#!/usr/bin/env python3
"""
PySCF反応エネルギー計算
結合解離エネルギー、反応エンタルピー、ギブズエネルギー変化の計算

使用例:
# 酢酸の解離: CH3COOH → CH3COO- + H+
python calculate_reaction_energy.py --reactants "CC(=O)O" --products "CC(=O)[O-]" "[H+]" --charges "0" "-1,1"

# メタンの燃焼: CH4 + 2O2 → CO2 + 2H2O
python calculate_reaction_energy.py --reactants "C" "O=O,O=O" --products "O=C=O" "O,O"
"""

import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pyscf import gto, scf, dft, hessian
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo
import warnings
warnings.filterwarnings('ignore')

def smiles_to_xyz(smiles):
    """SMILESから3D座標を生成"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    
    # 小分子の場合、MMFFが失敗することがあるので例外処理
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    
    conf = mol.GetConformer()
    atoms = []
    coords = []
    
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(atom.GetSymbol())
        coords.append([pos.x, pos.y, pos.z])
    
    return atoms, np.array(coords), mol

def create_pyscf_mol(atoms, coords, basis='6-31+G*', charge=0, spin=0):
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

def calculate_single_molecule(smiles, method='B3LYP', basis='6-31+G*', 
                            charge=0, spin=0, optimize_geom=True, calc_freq=True):
    """単一分子の完全計算"""
    print(f"\n計算中: {smiles} (電荷={charge}, スピン={spin})")
    
    # 特殊なケース処理
    if smiles == "[H+]":
        # プロトンは電子を持たないので、エネルギー = 0
        return {
            'smiles': smiles,
            'energy': 0.0,
            'zpe': 0.0,
            'thermal_correction': 0.0,
            'enthalpy': 0.0,
            'gibbs': 0.0,
            'formula': 'H+',
            'special': True
        }
    elif smiles == "[H]" or smiles == "[H.]":
        # 水素原子
        mol = gto.M(atom='H 0 0 0', basis=basis, charge=0, spin=1)
        mf = scf.UHF(mol)
        energy = mf.kernel()
        return {
            'smiles': smiles,
            'energy': energy,
            'zpe': 0.0,
            'thermal_correction': 0.0,
            'enthalpy': energy,
            'gibbs': energy,
            'formula': 'H',
            'special': True
        }
    
    # 通常の分子
    try:
        atoms, coords, rdkit_mol = smiles_to_xyz(smiles)
    except:
        print(f"警告: {smiles}の3D構造生成に失敗")
        return None
    
    # 分子式取得
    formula = Chem.rdMolDescriptors.CalcMolFormula(rdkit_mol)
    
    # PySCF分子作成
    mol = create_pyscf_mol(atoms, coords, basis, charge, spin)
    
    # SCF計算
    if method == 'HF':
        if spin == 0:
            mf = scf.RHF(mol)
        else:
            mf = scf.UHF(mol)
    else:
        if spin == 0:
            mf = dft.RKS(mol)
        else:
            mf = dft.UKS(mol)
        mf.xc = method
    
    # 構造最適化
    if optimize_geom:
        print("  構造最適化中...")
        try:
            mol_opt = optimize(mf, maxsteps=50)
            
            # 最適化後の再計算
            if method == 'HF':
                if spin == 0:
                    mf = scf.RHF(mol_opt)
                else:
                    mf = scf.UHF(mol_opt)
            else:
                if spin == 0:
                    mf = dft.RKS(mol_opt)
                else:
                    mf = dft.UKS(mol_opt)
                mf.xc = method
            
            mol = mol_opt
        except:
            print("  警告: 構造最適化が収束しませんでした")
    
    # エネルギー計算
    energy = mf.kernel()
    
    if not mf.converged:
        print(f"  警告: SCF計算が収束しませんでした")
    
    # 熱力学補正
    thermal_data = {
        'zpe': 0.0,
        'thermal_correction': 0.0,
        'enthalpy': energy,
        'gibbs': energy
    }
    
    if calc_freq:
        try:
            print("  振動数解析中...")
            # Hessian計算
            if isinstance(mf, (scf.rhf.RHF, scf.rohf.ROHF)):
                h = hessian.RHF(mf)
            elif isinstance(mf, scf.uhf.UHF):
                h = hessian.UHF(mf)
            elif isinstance(mf, dft.rks.RKS):
                h = hessian.RKS(mf)
            else:
                h = hessian.UKS(mf)
            
            hess = h.kernel()
            
            # 熱力学的補正を計算
            freq_info = thermo.harmonic_analysis(mol, hess)
            thermo_results = thermo.thermo(
                freq_info['freq_au'],
                298.15,  # 温度 (K)
                pressure=101325  # 圧力 (Pa)
            )
            
            thermal_data['zpe'] = thermo_results[0]
            thermal_data['thermal_correction'] = thermo_results[1]
            thermal_data['enthalpy'] = energy + thermo_results[2]
            thermal_data['gibbs'] = energy + thermo_results[3]
            
        except Exception as e:
            print(f"  警告: 振動数解析失敗: {e}")
    
    return {
        'smiles': smiles,
        'formula': formula,
        'energy': energy,
        'zpe': thermal_data['zpe'],
        'thermal_correction': thermal_data['thermal_correction'],
        'enthalpy': thermal_data['enthalpy'],
        'gibbs': thermal_data['gibbs'],
        'charge': charge,
        'spin': spin
    }

def parse_molecules_and_stoichiometry(smiles_list):
    """SMILES文字列から分子と化学量論係数を解析"""
    molecules = []
    stoichiometry = []
    
    for item in smiles_list:
        parts = item.split(',')
        for part in parts:
            # 係数があるかチェック（例: "2H2O" → 2, "H2O"）
            coeff = 1
            smiles = part
            
            # 簡易的な係数抽出（完全ではない）
            if part[0].isdigit():
                coeff = int(part[0])
                smiles = part[1:]
            
            molecules.append(smiles)
            stoichiometry.append(coeff)
    
    return molecules, stoichiometry

def calculate_reaction_energy(reactants_data, products_data, 
                             reactant_stoich, product_stoich):
    """反応エネルギーを計算"""
    
    # 電子エネルギー
    e_reactants = sum(coeff * data['energy'] 
                     for coeff, data in zip(reactant_stoich, reactants_data))
    e_products = sum(coeff * data['energy'] 
                    for coeff, data in zip(product_stoich, products_data))
    delta_e = e_products - e_reactants
    
    # ZPE補正エネルギー
    e0_reactants = sum(coeff * (data['energy'] + data['zpe']) 
                      for coeff, data in zip(reactant_stoich, reactants_data))
    e0_products = sum(coeff * (data['energy'] + data['zpe']) 
                     for coeff, data in zip(product_stoich, products_data))
    delta_e0 = e0_products - e0_reactants
    
    # エンタルピー
    h_reactants = sum(coeff * data['enthalpy'] 
                     for coeff, data in zip(reactant_stoich, reactants_data))
    h_products = sum(coeff * data['enthalpy'] 
                    for coeff, data in zip(product_stoich, products_data))
    delta_h = h_products - h_reactants
    
    # ギブズ自由エネルギー
    g_reactants = sum(coeff * data['gibbs'] 
                     for coeff, data in zip(reactant_stoich, reactants_data))
    g_products = sum(coeff * data['gibbs'] 
                    for coeff, data in zip(product_stoich, products_data))
    delta_g = g_products - g_reactants
    
    return {
        'delta_e': delta_e,
        'delta_e0': delta_e0,
        'delta_h': delta_h,
        'delta_g': delta_g
    }

def calculate_bond_dissociation_energy(parent_smiles, fragments_smiles, 
                                      method='B3LYP', basis='6-31+G*'):
    """結合解離エネルギーを計算"""
    print(f"\n結合解離エネルギー計算: {parent_smiles} → {' + '.join(fragments_smiles)}")
    
    # 親分子の計算
    parent_data = calculate_single_molecule(parent_smiles, method, basis)
    
    # フラグメントの計算（ラジカルの場合はスピンを設定）
    fragments_data = []
    for frag in fragments_smiles:
        # ラジカル判定（簡易版）
        spin = 1 if '.' in frag or '[' in frag else 0
        frag_data = calculate_single_molecule(frag, method, basis, spin=spin)
        fragments_data.append(frag_data)
    
    if parent_data and all(fragments_data):
        bde = sum(f['energy'] for f in fragments_data) - parent_data['energy']
        bde_zpe = sum(f['energy'] + f['zpe'] for f in fragments_data) - \
                  (parent_data['energy'] + parent_data['zpe'])
        
        return {
            'bde': bde,
            'bde_zpe': bde_zpe,
            'parent': parent_data,
            'fragments': fragments_data
        }
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description='PySCF反応エネルギー計算')
    parser.add_argument('--reactants', type=str, nargs='+', required=True,
                       help='反応物のSMILES（スペース区切り）')
    parser.add_argument('--products', type=str, nargs='+', required=True,
                       help='生成物のSMILES（スペース区切り）')
    parser.add_argument('--method', type=str, default='B3LYP',
                       choices=['HF', 'B3LYP', 'PBE', 'M06-2X'],
                       help='計算手法 (default: B3LYP)')
    parser.add_argument('--basis', type=str, default='6-31+G*',
                       help='基底関数 (default: 6-31+G*)')
    parser.add_argument('--charges', type=str, default='0',
                       help='電荷（反応物,生成物）例: "0" or "0,-1,1"')
    parser.add_argument('--spins', type=str, default='0',
                       help='スピン多重度-1（反応物,生成物）')
    parser.add_argument('--no-opt', action='store_true',
                       help='構造最適化をスキップ')
    parser.add_argument('--no-freq', action='store_true',
                       help='振動数解析をスキップ')
    parser.add_argument('--temperature', type=float, default=298.15,
                       help='温度 (K) (default: 298.15)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PySCF 反応エネルギー計算")
    print("=" * 60)
    print(f"手法: {args.method}/{args.basis}")
    print(f"温度: {args.temperature} K")
    
    # 反応式の表示
    reactants_str = ' + '.join(args.reactants)
    products_str = ' + '.join(args.products)
    print(f"\n反応: {reactants_str} → {products_str}")
    
    # 電荷とスピンの解析
    charges = args.charges.split(',')
    spins = args.spins.split(',')
    
    # デフォルト値で埋める
    n_molecules = len(args.reactants) + len(args.products)
    if len(charges) == 1:
        charges = [int(charges[0])] * n_molecules
    else:
        charges = [int(c) for c in charges]
    
    if len(spins) == 1:
        spins = [int(spins[0])] * n_molecules
    else:
        spins = [int(s) for s in spins]
    
    # 反応物の計算
    print("\n[1] 反応物の計算...")
    reactants_data = []
    for i, smiles in enumerate(args.reactants):
        charge = charges[i] if i < len(charges) else 0
        spin = spins[i] if i < len(spins) else 0
        
        data = calculate_single_molecule(
            smiles, args.method, args.basis, 
            charge, spin, 
            not args.no_opt, not args.no_freq
        )
        
        if data:
            reactants_data.append(data)
            print(f"  {data['formula']}: E = {data['energy']:.6f} Hartree")
        else:
            print(f"  エラー: {smiles}の計算に失敗")
            return
    
    # 生成物の計算
    print("\n[2] 生成物の計算...")
    products_data = []
    offset = len(args.reactants)
    for i, smiles in enumerate(args.products):
        charge = charges[offset + i] if offset + i < len(charges) else 0
        spin = spins[offset + i] if offset + i < len(spins) else 0
        
        data = calculate_single_molecule(
            smiles, args.method, args.basis, 
            charge, spin,
            not args.no_opt, not args.no_freq
        )
        
        if data:
            products_data.append(data)
            print(f"  {data['formula']}: E = {data['energy']:.6f} Hartree")
        else:
            print(f"  エラー: {smiles}の計算に失敗")
            return
    
    # 反応エネルギー計算
    print("\n[3] 反応エネルギー計算...")
    
    # 化学量論係数（簡易版：すべて1）
    reactant_stoich = [1] * len(reactants_data)
    product_stoich = [1] * len(products_data)
    
    reaction_energy = calculate_reaction_energy(
        reactants_data, products_data,
        reactant_stoich, product_stoich
    )
    
    # 結果表示
    print("\n" + "=" * 60)
    print("反応エネルギー解析結果")
    print("=" * 60)
    
    print(f"\n【電子エネルギー変化】")
    print(f"ΔE = {reaction_energy['delta_e']:.6f} Hartree")
    print(f"    = {reaction_energy['delta_e']*627.509:.2f} kcal/mol")
    print(f"    = {reaction_energy['delta_e']*2625.5:.1f} kJ/mol")
    
    if not args.no_freq:
        print(f"\n【ゼロ点補正エネルギー変化】")
        print(f"ΔE₀ = {reaction_energy['delta_e0']:.6f} Hartree")
        print(f"     = {reaction_energy['delta_e0']*627.509:.2f} kcal/mol")
        print(f"     = {reaction_energy['delta_e0']*2625.5:.1f} kJ/mol")
        
        print(f"\n【エンタルピー変化】(T = {args.temperature} K)")
        print(f"ΔH = {reaction_energy['delta_h']:.6f} Hartree")
        print(f"    = {reaction_energy['delta_h']*627.509:.2f} kcal/mol")
        print(f"    = {reaction_energy['delta_h']*2625.5:.1f} kJ/mol")
        
        print(f"\n【ギブズ自由エネルギー変化】(T = {args.temperature} K)")
        print(f"ΔG = {reaction_energy['delta_g']:.6f} Hartree")
        print(f"    = {reaction_energy['delta_g']*627.509:.2f} kcal/mol")
        print(f"    = {reaction_energy['delta_g']*2625.5:.1f} kJ/mol")
        
        # 平衡定数の推定
        R = 8.314  # J/(mol·K)
        delta_g_joule = reaction_energy['delta_g'] * 2625500  # J/mol
        K_eq = np.exp(-delta_g_joule / (R * args.temperature))
        print(f"\n平衡定数 K = {K_eq:.2e}")
        
        if K_eq > 1e10:
            print("  → 強く生成物側に偏る（ほぼ完全反応）")
        elif K_eq > 1:
            print("  → 生成物側に偏る")
        elif K_eq > 1e-10:
            print("  → 反応物側に偏る")
        else:
            print("  → 強く反応物側に偏る（ほとんど反応しない）")
    
    # 反応の熱力学的評価
    print(f"\n【反応の評価】")
    if reaction_energy['delta_e'] < 0:
        print("✅ 発熱反応（エネルギー的に有利）")
    else:
        print("⚠️ 吸熱反応（エネルギー的に不利）")
    
    if not args.no_freq and reaction_energy['delta_g'] < 0:
        print("✅ 自発的反応（ΔG < 0）")
    elif not args.no_freq:
        print("⚠️ 非自発的反応（ΔG > 0）")
    
    # 個別分子のエネルギー詳細
    print("\n" + "=" * 60)
    print("分子エネルギー詳細")
    print("=" * 60)
    
    print("\n反応物:")
    for data in reactants_data:
        print(f"  {data['formula']}:")
        print(f"    電子エネルギー: {data['energy']:.6f} Hartree")
        if not args.no_freq:
            print(f"    ZPE: {data['zpe']*627.509:.2f} kcal/mol")
            print(f"    G: {data['gibbs']:.6f} Hartree")
    
    print("\n生成物:")
    for data in products_data:
        print(f"  {data['formula']}:")
        print(f"    電子エネルギー: {data['energy']:.6f} Hartree")
        if not args.no_freq:
            print(f"    ZPE: {data['zpe']*627.509:.2f} kcal/mol")
            print(f"    G: {data['gibbs']:.6f} Hartree")
    
    # 結果をファイルに保存
    output_file = "reaction_energy_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"Reaction: {reactants_str} → {products_str}\n")
        f.write(f"Method: {args.method}/{args.basis}\n")
        f.write(f"Temperature: {args.temperature} K\n\n")
        
        f.write("Energy Changes:\n")
        f.write(f"ΔE: {reaction_energy['delta_e']*627.509:.2f} kcal/mol\n")
        if not args.no_freq:
            f.write(f"ΔE₀: {reaction_energy['delta_e0']*627.509:.2f} kcal/mol\n")
            f.write(f"ΔH: {reaction_energy['delta_h']*627.509:.2f} kcal/mol\n")
            f.write(f"ΔG: {reaction_energy['delta_g']*627.509:.2f} kcal/mol\n")
            f.write(f"K_eq: {K_eq:.2e}\n")
    
    print(f"\n結果を {output_file} に保存しました")
    
    print("\n" + "=" * 60)
    print("計算完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()