#!/usr/bin/env python3
"""
PySCF分子間相互作用エネルギー計算
水素結合、π-π相互作用、van der Waals相互作用の解析

使用例:
# 水二量体の水素結合
python calculate_interaction.py --molecule1 "O" --molecule2 "O" --distance 2.8

# ベンゼン二量体のπ-π相互作用
python calculate_interaction.py --molecule1 "c1ccccc1" --molecule2 "c1ccccc1" --mode "parallel" --scan
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pyscf import gto, scf, dft, mp
from pyscf.geomopt.geometric_solver import optimize
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

def create_dimer_geometry(atoms1, coords1, atoms2, coords2, 
                         distance=3.5, mode='aligned'):
    """二量体の幾何構造を作成"""
    
    # 分子2を移動
    if mode == 'aligned':
        # x軸方向に並べる
        offset = np.array([distance, 0, 0])
    elif mode == 'parallel':
        # z軸方向に平行配置（π-π相互作用）
        offset = np.array([0, 0, distance])
    elif mode == 'perpendicular':
        # T字型配置
        offset = np.array([0, distance, 0])
    else:
        offset = np.array([distance, 0, 0])
    
    # 分子1の重心
    com1 = np.mean(coords1, axis=0)
    # 分子2の重心
    com2 = np.mean(coords2, axis=0)
    
    # 分子2を移動
    coords2_moved = coords2 - com2 + com1 + offset
    
    # 結合
    atoms_dimer = atoms1 + atoms2
    coords_dimer = np.vstack([coords1, coords2_moved])
    
    return atoms_dimer, coords_dimer

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

def calculate_energy(mol, method='B3LYP', use_bsse_correction=False):
    """エネルギー計算"""
    
    if method == 'HF':
        mf = scf.RHF(mol)
    elif method == 'MP2':
        mf = scf.RHF(mol)
        mf.kernel()
        mp2 = mp.MP2(mf)
        e_corr, t2 = mp2.kernel()
        return mf.e_tot + e_corr
    elif method in ['B3LYP', 'PBE', 'M06-2X', 'B3LYP-D3']:
        mf = dft.RKS(mol)
        if method == 'B3LYP-D3':
            mf.xc = 'B3LYP'
            # D3分散補正（簡易版）
            # 実際にはdftd3ライブラリが必要
        else:
            mf.xc = method
    else:
        raise ValueError(f"Unknown method: {method}")
    
    energy = mf.kernel()
    
    if not mf.converged:
        print(f"  警告: SCF計算が収束しませんでした")
    
    return energy

def calculate_bsse_correction(atoms1, coords1, atoms2, coords2, 
                             atoms_dimer, coords_dimer, method, basis):
    """BSSE (Basis Set Superposition Error) 補正を計算"""
    print("  BSSE補正計算中...")
    
    # ゴースト原子を使った計算
    n_atoms1 = len(atoms1)
    
    # 分子1 + ゴースト分子2
    ghost_atoms2 = ['Ghost:' + atom for atom in atoms2]
    atoms_1_ghost2 = atoms1 + ghost_atoms2
    mol_1_ghost2 = create_pyscf_mol(atoms_1_ghost2, coords_dimer, basis)
    e_1_ghost2 = calculate_energy(mol_1_ghost2, method)
    
    # ゴースト分子1 + 分子2
    ghost_atoms1 = ['Ghost:' + atom for atom in atoms1]
    atoms_ghost1_2 = ghost_atoms1 + atoms2
    mol_ghost1_2 = create_pyscf_mol(atoms_ghost1_2, coords_dimer, basis)
    e_ghost1_2 = calculate_energy(mol_ghost1_2, method)
    
    # 分子1のみ（二量体基底）
    mol_1_only = create_pyscf_mol(atoms1, coords1, basis)
    e_1_only = calculate_energy(mol_1_only, method)
    
    # 分子2のみ（二量体基底）
    mol_2_only = create_pyscf_mol(atoms2, coords2, basis)
    e_2_only = calculate_energy(mol_2_only, method)
    
    # BSSE補正
    bsse = (e_1_ghost2 - e_1_only) + (e_ghost1_2 - e_2_only)
    
    return bsse

def calculate_interaction_energy(mol1_smiles, mol2_smiles, distance, 
                                mode='aligned', method='B3LYP', basis='6-31+G*',
                                use_bsse=True, optimize_dimer=False):
    """相互作用エネルギーを計算"""
    
    # 分子1の構造
    atoms1, coords1 = smiles_to_xyz(mol1_smiles)
    mol1 = create_pyscf_mol(atoms1, coords1, basis)
    
    # 分子2の構造
    atoms2, coords2 = smiles_to_xyz(mol2_smiles)
    mol2 = create_pyscf_mol(atoms2, coords2, basis)
    
    # 二量体の構造
    atoms_dimer, coords_dimer = create_dimer_geometry(
        atoms1, coords1, atoms2, coords2, distance, mode
    )
    mol_dimer = create_pyscf_mol(atoms_dimer, coords_dimer, basis)
    
    # 二量体の最適化（オプション）
    if optimize_dimer:
        print("  二量体構造最適化中...")
        mf_dimer = dft.RKS(mol_dimer)
        mf_dimer.xc = method if method != 'HF' else 'B3LYP'
        mol_dimer_opt = optimize(mf_dimer, maxsteps=30)
        coords_dimer = mol_dimer_opt.atom_coords() * 0.529177  # Bohr to Angstrom
        mol_dimer = mol_dimer_opt
    
    # エネルギー計算
    print(f"  距離 {distance:.2f} Åでの計算...")
    e_mol1 = calculate_energy(mol1, method)
    e_mol2 = calculate_energy(mol2, method)
    e_dimer = calculate_energy(mol_dimer, method)
    
    # 相互作用エネルギー
    e_int = e_dimer - e_mol1 - e_mol2
    
    # BSSE補正
    bsse = 0.0
    if use_bsse:
        try:
            n_atoms1 = len(atoms1)
            coords1_in_dimer = coords_dimer[:n_atoms1]
            coords2_in_dimer = coords_dimer[n_atoms1:]
            
            bsse = calculate_bsse_correction(
                atoms1, coords1_in_dimer, 
                atoms2, coords2_in_dimer,
                atoms_dimer, coords_dimer, 
                method, basis
            )
            print(f"  BSSE補正: {bsse*627.509:.3f} kcal/mol")
        except:
            print("  警告: BSSE補正計算に失敗")
            bsse = 0.0
    
    e_int_corrected = e_int - bsse
    
    return {
        'distance': distance,
        'e_mol1': e_mol1,
        'e_mol2': e_mol2,
        'e_dimer': e_dimer,
        'e_int': e_int,
        'bsse': bsse,
        'e_int_corrected': e_int_corrected,
        'coords_dimer': coords_dimer
    }

def scan_interaction_energy(mol1_smiles, mol2_smiles, 
                           distances=None, mode='aligned', 
                           method='B3LYP', basis='6-31+G*'):
    """距離をスキャンして相互作用エネルギーカーブを計算"""
    
    if distances is None:
        distances = np.arange(2.0, 8.0, 0.5)
    
    results = []
    
    print(f"\n距離スキャン: {distances[0]:.1f} - {distances[-1]:.1f} Å")
    
    for dist in distances:
        result = calculate_interaction_energy(
            mol1_smiles, mol2_smiles, dist, mode, method, basis, 
            use_bsse=True, optimize_dimer=False
        )
        results.append(result)
        
        print(f"  d = {dist:.2f} Å: "
              f"E_int = {result['e_int']*627.509:.3f} kcal/mol, "
              f"E_int(BSSE補正) = {result['e_int_corrected']*627.509:.3f} kcal/mol")
    
    return results

def analyze_interaction_type(e_int_kcalmol, distance):
    """相互作用の種類を推定"""
    
    if e_int_kcalmol < -20:
        return "強いイオン結合/共有結合的相互作用"
    elif e_int_kcalmol < -10:
        return "強い水素結合"
    elif e_int_kcalmol < -5:
        return "中程度の水素結合"
    elif e_int_kcalmol < -2:
        return "弱い水素結合/強いvan der Waals相互作用"
    elif e_int_kcalmol < -0.5:
        return "van der Waals相互作用"
    elif e_int_kcalmol < 0:
        return "非常に弱い引力的相互作用"
    else:
        return "反発的相互作用"

def plot_interaction_curve(results, formula1, formula2, save_file=None):
    """相互作用エネルギーカーブをプロット"""
    
    distances = [r['distance'] for r in results]
    e_int = [r['e_int']*627.509 for r in results]  # kcal/mol
    e_int_corrected = [r['e_int_corrected']*627.509 for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # エネルギーカーブ
    ax.plot(distances, e_int, 'b-', linewidth=2, label='相互作用エネルギー')
    ax.plot(distances, e_int_corrected, 'r--', linewidth=2, label='BSSE補正後')
    
    # ゼロライン
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # 最小値
    min_idx = np.argmin(e_int_corrected)
    min_dist = distances[min_idx]
    min_energy = e_int_corrected[min_idx]
    
    ax.plot(min_dist, min_energy, 'ro', markersize=8)
    ax.annotate(f'最小値\n({min_dist:.2f} Å, {min_energy:.2f} kcal/mol)',
                xy=(min_dist, min_energy), xytext=(min_dist+0.5, min_energy-1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10)
    
    ax.set_xlabel('Distance (Å)', fontsize=12)
    ax.set_ylabel('Interaction Energy (kcal/mol)', fontsize=12)
    ax.set_title(f'{formula1} - {formula2} Interaction Energy', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"グラフを {save_file} に保存しました")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='PySCF分子間相互作用エネルギー計算')
    parser.add_argument('--molecule1', type=str, required=True,
                       help='分子1のSMILES')
    parser.add_argument('--molecule2', type=str, required=True,
                       help='分子2のSMILES')
    parser.add_argument('--distance', type=float, default=3.5,
                       help='分子間距離 (Å) (default: 3.5)')
    parser.add_argument('--mode', type=str, default='aligned',
                       choices=['aligned', 'parallel', 'perpendicular'],
                       help='配置モード (default: aligned)')
    parser.add_argument('--method', type=str, default='B3LYP',
                       choices=['HF', 'B3LYP', 'PBE', 'M06-2X', 'MP2'],
                       help='計算手法 (default: B3LYP)')
    parser.add_argument('--basis', type=str, default='6-31+G*',
                       help='基底関数 (default: 6-31+G*)')
    parser.add_argument('--scan', action='store_true',
                       help='距離スキャンを実行')
    parser.add_argument('--scan-range', type=str, default='2.0,8.0,0.5',
                       help='スキャン範囲 "開始,終了,ステップ"')
    parser.add_argument('--optimize-dimer', action='store_true',
                       help='二量体構造を最適化')
    parser.add_argument('--no-bsse', action='store_true',
                       help='BSSE補正をスキップ')
    parser.add_argument('--plot', action='store_true',
                       help='相互作用カーブをプロット')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PySCF 分子間相互作用エネルギー計算")
    print("=" * 60)
    print(f"分子1: {args.molecule1}")
    print(f"分子2: {args.molecule2}")
    print(f"手法: {args.method}/{args.basis}")
    print(f"配置: {args.mode}")
    
    # 分子情報取得
    mol1_rdkit = Chem.MolFromSmiles(args.molecule1)
    mol2_rdkit = Chem.MolFromSmiles(args.molecule2)
    formula1 = Chem.rdMolDescriptors.CalcMolFormula(mol1_rdkit)
    formula2 = Chem.rdMolDescriptors.CalcMolFormula(mol2_rdkit)
    print(f"分子式: {formula1} + {formula2}")
    
    if args.scan:
        # 距離スキャン
        print("\n[1] 距離スキャン計算...")
        scan_parts = args.scan_range.split(',')
        start = float(scan_parts[0])
        end = float(scan_parts[1])
        step = float(scan_parts[2])
        distances = np.arange(start, end + step/2, step)
        
        results = scan_interaction_energy(
            args.molecule1, args.molecule2,
            distances, args.mode, args.method, args.basis
        )
        
        # 最適距離を見つける
        e_int_corrected = [r['e_int_corrected']*627.509 for r in results]
        min_idx = np.argmin(e_int_corrected)
        optimal_dist = results[min_idx]['distance']
        min_energy = e_int_corrected[min_idx]
        
        print("\n" + "=" * 60)
        print("スキャン結果サマリー")
        print("=" * 60)
        print(f"\n最適距離: {optimal_dist:.2f} Å")
        print(f"最小相互作用エネルギー: {min_energy:.3f} kcal/mol")
        print(f"相互作用の種類: {analyze_interaction_type(min_energy, optimal_dist)}")
        
        # 結合エネルギー（最小値の絶対値）
        if min_energy < 0:
            print(f"結合エネルギー: {abs(min_energy):.3f} kcal/mol")
        
        # グラフ作成
        if args.plot:
            plot_file = f"{formula1}_{formula2}_interaction.png"
            plot_interaction_curve(results, formula1, formula2, plot_file)
        
        # 詳細データ保存
        output_file = f"{formula1}_{formula2}_scan.txt"
        with open(output_file, 'w') as f:
            f.write(f"# {formula1} - {formula2} Interaction Energy Scan\n")
            f.write(f"# Method: {args.method}/{args.basis}\n")
            f.write(f"# Mode: {args.mode}\n\n")
            f.write(f"{'Distance (Å)':^12} {'E_int (kcal/mol)':^18} "
                   f"{'E_int_corrected':^18} {'BSSE':^12}\n")
            f.write("-" * 60 + "\n")
            
            for r in results:
                f.write(f"{r['distance']:^12.2f} "
                       f"{r['e_int']*627.509:^18.3f} "
                       f"{r['e_int_corrected']*627.509:^18.3f} "
                       f"{r['bsse']*627.509:^12.3f}\n")
        
        print(f"\n詳細データを {output_file} に保存しました")
        
    else:
        # 単一距離での計算
        print(f"\n[1] 距離 {args.distance} Åでの計算...")
        
        result = calculate_interaction_energy(
            args.molecule1, args.molecule2,
            args.distance, args.mode, args.method, args.basis,
            use_bsse=not args.no_bsse,
            optimize_dimer=args.optimize_dimer
        )
        
        # 結果表示
        print("\n" + "=" * 60)
        print("相互作用エネルギー解析")
        print("=" * 60)
        
        print(f"\n【エネルギー成分】")
        print(f"分子1: {result['e_mol1']:.6f} Hartree")
        print(f"分子2: {result['e_mol2']:.6f} Hartree")
        print(f"二量体: {result['e_dimer']:.6f} Hartree")
        
        print(f"\n【相互作用エネルギー】")
        print(f"E_int = E(二量体) - E(分子1) - E(分子2)")
        print(f"     = {result['e_int']:.6f} Hartree")
        print(f"     = {result['e_int']*627.509:.3f} kcal/mol")
        print(f"     = {result['e_int']*2625.5:.1f} kJ/mol")
        
        if not args.no_bsse:
            print(f"\nBSSE補正: {result['bsse']*627.509:.3f} kcal/mol")
            print(f"補正後: {result['e_int_corrected']*627.509:.3f} kcal/mol")
        
        # 相互作用の分類
        e_int_kcal = result['e_int_corrected']*627.509
        interaction_type = analyze_interaction_type(e_int_kcal, args.distance)
        print(f"\n【相互作用の評価】")
        print(f"種類: {interaction_type}")
        
        if e_int_kcal < 0:
            print(f"✅ 引力的相互作用（安定化）")
        else:
            print(f"⚠️ 反発的相互作用（不安定化）")
        
        # 二量体構造を保存
        dimer_xyz_file = f"{formula1}_{formula2}_dimer.xyz"
        atoms1, coords1 = smiles_to_xyz(args.molecule1)
        atoms2, coords2 = smiles_to_xyz(args.molecule2)
        atoms_dimer, _ = create_dimer_geometry(
            atoms1, coords1, atoms2, coords2, args.distance, args.mode
        )
        
        with open(dimer_xyz_file, 'w') as f:
            f.write(f"{len(atoms_dimer)}\n")
            f.write(f"Dimer at {args.distance} Å\n")
            for atom, coord in zip(atoms_dimer, result['coords_dimer']):
                f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
        
        print(f"\n二量体構造を {dimer_xyz_file} に保存しました")
    
    print("\n" + "=" * 60)
    print("計算完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()