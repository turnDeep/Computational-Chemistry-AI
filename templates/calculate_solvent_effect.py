#!/usr/bin/env python3
"""
PySCF溶媒効果計算（PCMモデル）
気相と溶媒中でのエネルギー比較と溶媒和エネルギー計算

使用例:
python calculate_solvent_effect.py --smiles "CC(=O)O" --solvent water
python calculate_solvent_effect.py --smiles "c1ccccc1" --solvent benzene
"""

import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pyscf import gto, scf, dft, solvent
from pyscf.geomopt.geometric_solver import optimize
import torch
import warnings
warnings.filterwarnings('ignore')

# 溶媒パラメータ（誘電率）
SOLVENT_PARAMS = {
    'water': {'eps': 78.39, 'name': '水'},
    'methanol': {'eps': 32.70, 'name': 'メタノール'},
    'ethanol': {'eps': 24.55, 'name': 'エタノール'},
    'acetone': {'eps': 20.70, 'name': 'アセトン'},
    'dmso': {'eps': 46.70, 'name': 'DMSO'},
    'dmf': {'eps': 36.71, 'name': 'DMF'},
    'chloroform': {'eps': 4.89, 'name': 'クロロホルム'},
    'dichloromethane': {'eps': 8.93, 'name': 'ジクロロメタン'},
    'thf': {'eps': 7.58, 'name': 'THF'},
    'benzene': {'eps': 2.27, 'name': 'ベンゼン'},
    'toluene': {'eps': 2.38, 'name': 'トルエン'},
    'hexane': {'eps': 1.88, 'name': 'ヘキサン'},
}

def smiles_to_xyz(smiles):
    """SMILESから3D座標を生成"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    
    conf = mol.GetConformer()
    atoms = []
    coords = []
    
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(atom.GetSymbol())
        coords.append([pos.x, pos.y, pos.z])
    
    return atoms, np.array(coords)

def create_pyscf_mol(atoms, coords, basis='6-31G*', charge=0, spin=0):
    """PySCF分子オブジェクトを作成"""
    atom_str = ""
    for atom, coord in zip(atoms, coords):
        atom_str += f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}; "
    
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.unit = 'Angstrom'
    mol.build()
    
    return mol

def perform_gas_phase(mol, method='B3LYP', optimize_geom=True):
    """気相計算"""
    print("気相計算中...")
    
    if method == 'HF':
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = method
    
    if optimize_geom:
        print("  構造最適化中...")
        mol_opt = optimize(mf, maxsteps=30)
        
        # 最適化後の再計算
        if method == 'HF':
            mf = scf.RHF(mol_opt)
        else:
            mf = dft.RKS(mol_opt)
            mf.xc = method
        
        energy = mf.kernel()
        return mf, mol_opt, energy
    else:
        energy = mf.kernel()
        return mf, mol, energy

def perform_pcm_calculation(mol, method='B3LYP', solvent_name='water', optimize_geom=True):
    """PCM溶媒効果計算"""
    solvent_info = SOLVENT_PARAMS.get(solvent_name, SOLVENT_PARAMS['water'])
    print(f"{solvent_info['name']}中の計算 (ε = {solvent_info['eps']})...")
    
    if method == 'HF':
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = method
    
    # PCMモデルを適用
    mf = solvent.PCM(mf)
    mf.eps = solvent_info['eps']
    
    if optimize_geom:
        print("  溶媒中での構造最適化中...")
        mol_opt = optimize(mf, maxsteps=30)
        
        # 最適化後の再計算
        if method == 'HF':
            mf = scf.RHF(mol_opt)
        else:
            mf = dft.RKS(mol_opt)
            mf.xc = method
        
        mf = solvent.PCM(mf)
        mf.eps = solvent_info['eps']
        energy = mf.kernel()
        
        return mf, mol_opt, energy
    else:
        energy = mf.kernel()
        return mf, mol, energy

def calculate_dipole_moment(mf):
    """双極子モーメント計算"""
    dm = mf.make_rdm1()
    dipole = mf.dip_moment(mf.mol, dm, unit='Debye')
    return dipole, np.linalg.norm(dipole)

def analyze_orbitals(mf):
    """HOMO-LUMO解析"""
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    
    homo_idx = np.where(mo_occ > 0)[0][-1]
    lumo_idx = homo_idx + 1
    
    homo_energy = mo_energy[homo_idx]
    lumo_energy = mo_energy[lumo_idx] if lumo_idx < len(mo_energy) else None
    gap = lumo_energy - homo_energy if lumo_energy else None
    
    return homo_energy, lumo_energy, gap

def calculate_solvation_properties(e_gas, e_solv, mf_gas, mf_solv):
    """溶媒和特性を計算"""
    # 溶媒和エネルギー
    solvation_energy = e_solv - e_gas
    
    # 双極子モーメント
    dipole_gas, dipole_mag_gas = calculate_dipole_moment(mf_gas)
    dipole_solv, dipole_mag_solv = calculate_dipole_moment(mf_solv)
    
    # HOMO-LUMO
    homo_gas, lumo_gas, gap_gas = analyze_orbitals(mf_gas)
    homo_solv, lumo_solv, gap_solv = analyze_orbitals(mf_solv)
    
    return {
        'solvation_energy': solvation_energy,
        'dipole_gas': dipole_mag_gas,
        'dipole_solv': dipole_mag_solv,
        'dipole_change': dipole_mag_solv - dipole_mag_gas,
        'homo_gas': homo_gas,
        'homo_solv': homo_solv,
        'lumo_gas': lumo_gas,
        'lumo_solv': lumo_solv,
        'gap_gas': gap_gas,
        'gap_solv': gap_solv,
    }

def compare_multiple_solvents(mol, method='B3LYP', solvents=['water', 'ethanol', 'benzene']):
    """複数溶媒での比較"""
    results = {}
    
    # 気相計算（基準）
    mf_gas, mol_gas, e_gas = perform_gas_phase(mol, method, optimize_geom=False)
    
    for solvent_name in solvents:
        mf_solv, mol_solv, e_solv = perform_pcm_calculation(
            mol, method, solvent_name, optimize_geom=False
        )
        
        props = calculate_solvation_properties(e_gas, e_solv, mf_gas, mf_solv)
        props['energy_solv'] = e_solv
        props['solvent_eps'] = SOLVENT_PARAMS[solvent_name]['eps']
        results[solvent_name] = props
    
    results['gas'] = {
        'energy': e_gas,
        'dipole': calculate_dipole_moment(mf_gas)[1],
        'homo': analyze_orbitals(mf_gas)[0],
        'lumo': analyze_orbitals(mf_gas)[1],
        'gap': analyze_orbitals(mf_gas)[2],
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='PySCF溶媒効果計算（PCM）')
    parser.add_argument('--smiles', type=str, required=True,
                       help='計算対象分子のSMILES')
    parser.add_argument('--solvent', type=str, default='water',
                       choices=list(SOLVENT_PARAMS.keys()),
                       help='溶媒 (default: water)')
    parser.add_argument('--method', type=str, default='B3LYP',
                       choices=['HF', 'B3LYP', 'PBE', 'M06-2X'],
                       help='計算手法 (default: B3LYP)')
    parser.add_argument('--basis', type=str, default='6-31G*',
                       help='基底関数 (default: 6-31G*)')
    parser.add_argument('--charge', type=int, default=0,
                       help='分子の電荷 (default: 0)')
    parser.add_argument('--spin', type=int, default=0,
                       help='スピン多重度-1 (default: 0)')
    parser.add_argument('--optimize', action='store_true',
                       help='構造最適化を実行')
    parser.add_argument('--compare-solvents', action='store_true',
                       help='複数溶媒で比較')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PySCF 溶媒効果計算 (PCMモデル)")
    print("=" * 60)
    print(f"SMILES: {args.smiles}")
    print(f"手法: {args.method}/{args.basis}")
    
    # 分子情報取得
    mol_rdkit = Chem.MolFromSmiles(args.smiles)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol_rdkit)
    mw = Descriptors.MolWt(mol_rdkit)
    logp = Descriptors.MolLogP(mol_rdkit)
    print(f"分子式: {formula}")
    print(f"分子量: {mw:.2f} g/mol")
    print(f"LogP (推定): {logp:.2f}")
    
    # 3D構造生成
    print("\n[1] 3D構造生成...")
    atoms, coords = smiles_to_xyz(args.smiles)
    print(f"原子数: {len(atoms)}")
    
    # PySCF分子作成
    print("\n[2] PySCF分子オブジェクト作成...")
    mol = create_pyscf_mol(atoms, coords, args.basis, args.charge, args.spin)
    print(f"電子数: {mol.nelectron}")
    print(f"基底関数数: {mol.nao}")
    
    if args.compare_solvents:
        # 複数溶媒での比較
        print("\n[3] 複数溶媒での比較計算...")
        solvents_to_compare = ['water', 'methanol', 'ethanol', 'acetone', 
                              'chloroform', 'benzene', 'hexane']
        
        results = compare_multiple_solvents(mol, args.method, solvents_to_compare)
        
        print("\n" + "=" * 80)
        print(f"{'溶媒':^12} {'ε':^6} {'ΔGsolv (kcal/mol)':^18} "
              f"{'双極子 (D)':^12} {'HOMO-LUMO Gap (eV)':^18}")
        print("-" * 80)
        
        # 気相
        gas_res = results['gas']
        print(f"{'気相':^12} {'-':^6} {'-':^18} "
              f"{gas_res['dipole']:^12.2f} {gas_res['gap']*27.2114:^18.2f}")
        
        # 溶媒ごと
        sorted_solvents = sorted(solvents_to_compare, 
                               key=lambda x: SOLVENT_PARAMS[x]['eps'])
        
        for solvent_name in sorted_solvents:
            res = results[solvent_name]
            solv_info = SOLVENT_PARAMS[solvent_name]
            solvation_kcal = res['solvation_energy'] * 627.509
            
            print(f"{solv_info['name']:^12} {solv_info['eps']:^6.1f} "
                  f"{solvation_kcal:^18.2f} "
                  f"{res['dipole_solv']:^12.2f} "
                  f"{res['gap_solv']*27.2114:^18.2f}")
        
        print("=" * 80)
        
        # 最も安定な溶媒
        best_solvent = min(results.keys() - {'gas'}, 
                          key=lambda x: results[x]['solvation_energy'])
        best_energy = results[best_solvent]['solvation_energy'] * 627.509
        print(f"\n最も安定な溶媒: {SOLVENT_PARAMS[best_solvent]['name']} "
              f"(ΔGsolv = {best_energy:.2f} kcal/mol)")
        
    else:
        # 単一溶媒での詳細計算
        print(f"\n[3] 気相 vs {SOLVENT_PARAMS[args.solvent]['name']}の計算...")
        
        # 気相計算
        print("\n気相計算:")
        mf_gas, mol_gas, e_gas = perform_gas_phase(mol, args.method, args.optimize)
        print(f"  エネルギー: {e_gas:.6f} Hartree")
        
        # 溶媒中計算
        print(f"\n{SOLVENT_PARAMS[args.solvent]['name']}中:")
        mf_solv, mol_solv, e_solv = perform_pcm_calculation(
            mol, args.method, args.solvent, args.optimize
        )
        print(f"  エネルギー: {e_solv:.6f} Hartree")
        
        # 溶媒和特性計算
        print("\n[4] 溶媒効果の解析...")
        props = calculate_solvation_properties(e_gas, e_solv, mf_gas, mf_solv)
        
        print("\n" + "=" * 60)
        print("溶媒効果サマリー")
        print("=" * 60)
        
        # エネルギー
        print(f"\n【エネルギー】")
        print(f"気相:     {e_gas:.6f} Hartree")
        print(f"溶媒中:   {e_solv:.6f} Hartree")
        print(f"溶媒和エネルギー: {props['solvation_energy']:.6f} Hartree")
        print(f"                = {props['solvation_energy']*627.509:.2f} kcal/mol")
        print(f"                = {props['solvation_energy']*2625.5:.1f} kJ/mol")
        
        # 双極子モーメント
        print(f"\n【双極子モーメント】")
        print(f"気相:     {props['dipole_gas']:.3f} Debye")
        print(f"溶媒中:   {props['dipole_solv']:.3f} Debye")
        print(f"変化:     {props['dipole_change']:.3f} Debye "
              f"({props['dipole_change']/props['dipole_gas']*100:.1f}%)")
        
        # 分子軌道
        print(f"\n【分子軌道エネルギー】")
        print(f"HOMO (気相):   {props['homo_gas']:.4f} Hartree "
              f"({props['homo_gas']*27.2114:.2f} eV)")
        print(f"HOMO (溶媒):   {props['homo_solv']:.4f} Hartree "
              f"({props['homo_solv']*27.2114:.2f} eV)")
        
        if props['lumo_gas'] and props['lumo_solv']:
            print(f"LUMO (気相):   {props['lumo_gas']:.4f} Hartree "
                  f"({props['lumo_gas']*27.2114:.2f} eV)")
            print(f"LUMO (溶媒):   {props['lumo_solv']:.4f} Hartree "
                  f"({props['lumo_solv']*27.2114:.2f} eV)")
            print(f"Gap (気相):    {props['gap_gas']:.4f} Hartree "
                  f"({props['gap_gas']*27.2114:.2f} eV)")
            print(f"Gap (溶媒):    {props['gap_solv']:.4f} Hartree "
                  f"({props['gap_solv']*27.2114:.2f} eV)")
        
        # 安定化の判定
        print(f"\n【安定性評価】")
        if props['solvation_energy'] < 0:
            print(f"✅ 溶媒中で安定化 ({abs(props['solvation_energy']*627.509):.2f} kcal/mol)")
            if abs(props['solvation_energy']*627.509) > 10:
                print("   → 強い溶媒和（極性相互作用が強い）")
            elif abs(props['solvation_energy']*627.509) > 5:
                print("   → 中程度の溶媒和")
            else:
                print("   → 弱い溶媒和")
        else:
            print(f"⚠️ 溶媒中で不安定化 ({props['solvation_energy']*627.509:.2f} kcal/mol)")
        
        print("=" * 60)
    
    # 結果をファイルに保存
    output_file = f"{formula}_solvent_effect.txt"
    with open(output_file, 'w') as f:
        f.write(f"SMILES: {args.smiles}\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Method: {args.method}/{args.basis}\n")
        
        if args.compare_solvents:
            f.write("\nSolvation Energies (kcal/mol):\n")
            for solvent_name in sorted(results.keys() - {'gas'}, 
                                     key=lambda x: results[x]['solvation_energy']):
                if solvent_name != 'gas':
                    f.write(f"  {SOLVENT_PARAMS[solvent_name]['name']}: "
                           f"{results[solvent_name]['solvation_energy']*627.509:.2f}\n")
        else:
            f.write(f"\nSolvent: {SOLVENT_PARAMS[args.solvent]['name']}\n")
            f.write(f"Gas Phase Energy: {e_gas:.6f} Hartree\n")
            f.write(f"Solution Energy: {e_solv:.6f} Hartree\n")
            f.write(f"Solvation Energy: {props['solvation_energy']*627.509:.2f} kcal/mol\n")
    
    print(f"\n結果を {output_file} に保存しました")
    
    print("\n" + "=" * 60)
    print("計算完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()