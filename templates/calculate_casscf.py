#!/usr/bin/env python3
"""
PySCF CASSCF/CASCI計算
強相関系・多参照計算による高精度電子状態解析

使用例:
python calculate_casscf.py --smiles "O2" --active-space 2,2  # 酸素分子（三重項）
python calculate_casscf.py --smiles "c1ccccc1" --active-space 6,6  # ベンゼンのπ電子
"""

import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pyscf import gto, scf, mcscf, fci, tools
from pyscf.mcscf import avas
import warnings
warnings.filterwarnings('ignore')

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

def create_pyscf_mol(atoms, coords, basis='cc-pVDZ', charge=0, spin=0):
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

def perform_hf(mol):
    """Hartree-Fock計算（CASSCF初期推定用）"""
    print("Hartree-Fock計算中...")
    
    if mol.spin == 0:
        mf = scf.RHF(mol)
    else:
        mf = scf.ROHF(mol)
    
    mf.conv_tol = 1e-8
    energy = mf.kernel()
    
    if not mf.converged:
        print("⚠️ HF計算が収束しませんでした")
    
    return mf, energy

def select_active_space_avas(mol, mf, ao_labels=None, threshold=0.2):
    """AVAS法による活性空間の自動選択"""
    print("AVAS法による活性空間選択...")
    
    if ao_labels is None:
        # デフォルト: π軌道やd軌道を活性空間に
        if any(atom[0] in ['C', 'N', 'O'] for atom in mol._atom):
            ao_labels = ['C 2p', 'N 2p', 'O 2p']  # π系
        else:
            ao_labels = mol.search_ao_label('2p')  # p軌道
    
    # AVAS実行
    norb, ne_act, orbs = avas.avas(mf, ao_labels, threshold=threshold)
    
    print(f"AVAS選択: {ne_act}電子, {norb}軌道")
    
    return norb, ne_act, orbs

def perform_casscf(mol, mf, ncas, nelecas, nroots=1, state_average=False):
    """CASSCF計算"""
    print(f"\nCASSCF({nelecas},{ncas})計算中...")
    
    # CASSCF オブジェクト作成
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    
    # 状態平均CASSCF
    if state_average and nroots > 1:
        weights = np.ones(nroots) / nroots
        mc = mc.state_average(weights)
        print(f"状態平均CASSCF: {nroots}状態")
    
    # 収束条件
    mc.conv_tol = 1e-7
    mc.max_cycle_macro = 100
    
    # CASSCF実行
    try:
        e_casscf = mc.kernel()
        
        if not mc.converged:
            print("⚠️ CASSCF計算が収束しませんでした")
    except Exception as e:
        print(f"❌ CASSCF計算エラー: {e}")
        return None
    
    return mc

def perform_casci(mol, mf, ncas, nelecas, nroots=5):
    """CASCI計算（複数の電子状態）"""
    print(f"\nCASCI({nelecas},{ncas})計算中 ({nroots}状態)...")
    
    # CASCI オブジェクト作成
    mc = mcscf.CASCI(mf, ncas, nelecas)
    mc.fcisolver.nroots = nroots
    
    # CASCI実行
    try:
        e_casci, civec, ss, mocas = mc.kernel()
        
        # 複数状態の場合
        if isinstance(e_casci, np.ndarray):
            return mc, e_casci, civec
        else:
            return mc, [e_casci], [civec]
    except Exception as e:
        print(f"❌ CASCI計算エラー: {e}")
        return None, None, None

def analyze_casscf_results(mc):
    """CASSCF結果の解析"""
    results = {}
    
    # 基本情報
    results['energy'] = mc.e_tot
    results['e_cas'] = mc.e_cas  # CAS空間のエネルギー
    
    # 自然軌道占有数
    natocc = mc.mo_occ
    cas_occ = mc.mo_occ[mc.ncore:mc.ncore+mc.ncas]
    results['natural_occupations'] = cas_occ
    
    # エントロピー（多参照性の指標）
    entropy = -np.sum(cas_occ * np.log(cas_occ + 1e-14) + 
                     (2 - cas_occ) * np.log(2 - cas_occ + 1e-14))
    results['entropy'] = entropy
    
    # CI係数解析
    if hasattr(mc, 'ci'):
        ci_vec = mc.ci.flatten()
        max_coeff_idx = np.argmax(np.abs(ci_vec))
        results['max_ci_coefficient'] = ci_vec[max_coeff_idx]
        results['ci_weight'] = ci_vec[max_coeff_idx]**2
        
        # 主要な配置の数（|c| > 0.1）
        significant_configs = np.sum(np.abs(ci_vec) > 0.1)
        results['n_significant_configs'] = significant_configs
    
    return results

def analyze_excited_states(mc, e_states, civecs):
    """励起状態の解析"""
    print("\n励起状態解析:")
    print("-" * 60)
    print(f"{'State':^8} {'Energy (Hartree)':^18} {'Excitation (eV)':^16} {'<S^2>':^10}")
    print("-" * 60)
    
    results = []
    e_ground = e_states[0]
    
    for i, energy in enumerate(e_states):
        excitation_ev = (energy - e_ground) * 27.2114
        
        # スピン期待値計算
        ss_val = mc.fcisolver.spin_square(civecs[i], mc.ncas, mc.nelecas)[0]
        
        results.append({
            'state': i,
            'energy': energy,
            'excitation_ev': excitation_ev,
            'spin_squared': ss_val
        })
        
        state_type = "S₀" if i == 0 else f"S{i}" if ss_val < 0.1 else f"T{i}"
        print(f"{state_type:^8} {energy:^18.6f} {excitation_ev:^16.3f} {ss_val:^10.3f}")
    
    return results

def calculate_bond_order(mc):
    """結合次数の計算（自然軌道占有数から）"""
    # 簡易的な結合次数推定
    cas_occ = mc.mo_occ[mc.ncore:mc.ncore+mc.ncas]
    
    # 結合性・反結合性軌道の識別（簡易版）
    bonding = np.sum(cas_occ[cas_occ > 1.0])
    antibonding = np.sum(2.0 - cas_occ[cas_occ < 1.0])
    
    bond_order = (bonding - antibonding) / 2.0
    
    return bond_order

def save_molden_file(mol, mc, filename):
    """軌道をMolden形式で保存"""
    from pyscf import tools
    
    with open(filename, 'w') as f:
        tools.molden.header(mol, f)
        tools.molden.orbital_coeff(mol, f, mc.mo_coeff, 
                                   ene=mc.mo_energy, occ=mc.mo_occ)
    print(f"軌道を {filename} に保存しました")

def main():
    parser = argparse.ArgumentParser(description='PySCF CASSCF/CASCI計算')
    parser.add_argument('--smiles', type=str, required=True,
                       help='計算対象分子のSMILES')
    parser.add_argument('--active-space', type=str, default='auto',
                       help='活性空間 (電子数,軌道数) 例: 6,6 またはauto')
    parser.add_argument('--basis', type=str, default='cc-pVDZ',
                       help='基底関数 (default: cc-pVDZ)')
    parser.add_argument('--charge', type=int, default=0,
                       help='分子の電荷 (default: 0)')
    parser.add_argument('--spin', type=int, default=0,
                       help='スピン多重度-1 (default: 0)')
    parser.add_argument('--nroots', type=int, default=3,
                       help='計算する電子状態数 (default: 3)')
    parser.add_argument('--state-average', action='store_true',
                       help='状態平均CASSCFを実行')
    parser.add_argument('--casci-only', action='store_true',
                       help='CASCIのみ実行（軌道最適化なし）')
    parser.add_argument('--save-molden', action='store_true',
                       help='軌道をMolden形式で保存')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PySCF CASSCF/CASCI 強相関系計算")
    print("=" * 60)
    print(f"SMILES: {args.smiles}")
    print(f"基底関数: {args.basis}")
    
    # 分子情報取得
    mol_rdkit = Chem.MolFromSmiles(args.smiles)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol_rdkit)
    mw = Descriptors.MolWt(mol_rdkit)
    print(f"分子式: {formula}")
    print(f"分子量: {mw:.2f} g/mol")
    
    # 3D構造生成
    print("\n[1] 3D構造生成...")
    atoms, coords = smiles_to_xyz(args.smiles)
    print(f"原子数: {len(atoms)}")
    
    # PySCF分子作成
    print("\n[2] PySCF分子オブジェクト作成...")
    mol = create_pyscf_mol(atoms, coords, args.basis, args.charge, args.spin)
    print(f"電子数: {mol.nelectron}")
    print(f"基底関数数: {mol.nao}")
    
    # HF計算
    print("\n[3] Hartree-Fock計算...")
    mf, e_hf = perform_hf(mol)
    print(f"HFエネルギー: {e_hf:.6f} Hartree")
    
    # HOMO-LUMO
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    homo_idx = np.where(mo_occ > 0)[0][-1]
    lumo_idx = homo_idx + 1
    gap = mo_energy[lumo_idx] - mo_energy[homo_idx]
    print(f"HOMO-LUMO ギャップ: {gap*27.2114:.2f} eV")
    
    # 活性空間の決定
    print("\n[4] 活性空間の選択...")
    if args.active_space == 'auto':
        # AVAS法で自動選択
        ncas, nelecas, mo_coeff = select_active_space_avas(mol, mf)
        mf.mo_coeff = mo_coeff
    else:
        # 手動指定
        parts = args.active_space.split(',')
        nelecas = int(parts[0])
        ncas = int(parts[1])
        print(f"手動指定: {nelecas}電子, {ncas}軌道")
    
    # CASSCF/CASCI計算
    if args.casci_only:
        print("\n[5] CASCI計算...")
        mc, e_states, civecs = perform_casci(mol, mf, ncas, nelecas, args.nroots)
        
        if mc is not None:
            # 励起状態解析
            state_results = analyze_excited_states(mc, e_states, civecs)
            
            print(f"\n基底状態エネルギー:")
            print(f"  HF:    {e_hf:.6f} Hartree")
            print(f"  CASCI: {e_states[0]:.6f} Hartree")
            print(f"  相関エネルギー: {(e_states[0] - e_hf)*627.509:.2f} kcal/mol")
    else:
        print("\n[5] CASSCF計算...")
        mc = perform_casscf(mol, mf, ncas, nelecas, args.nroots, args.state_average)
        
        if mc is not None:
            # 結果解析
            results = analyze_casscf_results(mc)
            
            print("\n" + "=" * 60)
            print("CASSCF結果")
            print("=" * 60)
            
            print(f"\n【エネルギー】")
            print(f"HF エネルギー:      {e_hf:.6f} Hartree")
            print(f"CASSCF エネルギー:  {results['energy']:.6f} Hartree")
            print(f"相関エネルギー:     {(results['energy'] - e_hf)*627.509:.2f} kcal/mol")
            
            print(f"\n【活性空間】")
            print(f"電子数: {nelecas}")
            print(f"軌道数: {ncas}")
            print(f"CAS エネルギー: {results['e_cas']:.6f} Hartree")
            
            print(f"\n【自然軌道占有数】")
            nat_occ = results['natural_occupations']
            for i, occ in enumerate(nat_occ):
                orbital_type = "二重占有" if occ > 1.9 else "部分占有" if occ > 0.1 else "空軌道"
                print(f"  軌道 {i+1}: {occ:.4f} ({orbital_type})")
            
            print(f"\n【多参照性の指標】")
            print(f"エントロピー S: {results['entropy']:.4f}")
            if results['entropy'] < 0.5:
                print("  → 単一参照的（HFで十分な可能性）")
            elif results['entropy'] < 1.0:
                print("  → 弱い多参照性")
            else:
                print("  → 強い多参照性（CASSCF必須）")
            
            if 'n_significant_configs' in results:
                print(f"主要配置数 (|c| > 0.1): {results['n_significant_configs']}")
                print(f"最大CI係数: {results['max_ci_coefficient']:.4f}")
                print(f"主配置の重み: {results['ci_weight']*100:.1f}%")
            
            # 結合次数（該当する場合）
            if 'O' in atoms and atoms.count('O') == 2:  # O2分子の場合
                bond_order = calculate_bond_order(mc)
                print(f"\n【結合次数】")
                print(f"推定結合次数: {bond_order:.2f}")
            
            # 状態平均の結果
            if args.state_average and hasattr(mc, 'e_states'):
                print(f"\n【状態平均エネルギー】")
                for i, e in enumerate(mc.e_states):
                    print(f"  状態 {i}: {e:.6f} Hartree")
            
            # Moldenファイル保存
            if args.save_molden:
                molden_file = f"{formula}_casscf.molden"
                save_molden_file(mol, mc, molden_file)
    
    # 結果をファイルに保存
    output_file = f"{formula}_CASSCF_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"SMILES: {args.smiles}\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Basis: {args.basis}\n")
        f.write(f"Active Space: CAS({nelecas},{ncas})\n\n")
        f.write(f"HF Energy: {e_hf:.6f} Hartree\n")
        
        if not args.casci_only and mc is not None:
            f.write(f"CASSCF Energy: {results['energy']:.6f} Hartree\n")
            f.write(f"Correlation Energy: {(results['energy'] - e_hf)*627.509:.2f} kcal/mol\n")
            f.write(f"Entropy: {results['entropy']:.4f}\n\n")
            f.write("Natural Orbital Occupations:\n")
            for i, occ in enumerate(nat_occ):
                f.write(f"  Orbital {i+1}: {occ:.4f}\n")
        elif mc is not None:
            f.write(f"CASCI Energies:\n")
            for i, e in enumerate(e_states[:5]):
                f.write(f"  State {i}: {e:.6f} Hartree\n")
    
    print(f"\n結果を {output_file} に保存しました")
    
    print("\n" + "=" * 60)
    print("計算完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()