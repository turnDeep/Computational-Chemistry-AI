#!/usr/bin/env python3
"""
PySCF構造最適化と振動数計算スクリプト（最小機能版）
Usage: python opt-freq.py --smiles "CCO"
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo
from tqdm import tqdm
import time
import warnings
try:
    import gpu4pyscf
    warnings.warn("gpu4pyscf available.")
except ImportError:
    warnings.warn("gpu4pyscf not available.")
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

def create_mol(atoms, coords, basis='6-31+G**', charge=0, spin=0):
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

def visualize_molecule(atoms, coords, title=""):
    """3D分子構造を可視化"""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # 原子の色とサイズ
    colors = {'H': 'white', 'C': 'gray', 'O': 'red', 'N': 'blue', 
              'F': 'green', 'Cl': 'green', 'Br': 'brown', 'S': 'yellow'}
    sizes = {'H': 50, 'C': 100, 'O': 100, 'N': 100, 
             'F': 80, 'Cl': 120, 'Br': 140, 'S': 120}
    
    for atom, coord in zip(atoms, coords):
        color = colors.get(atom, 'gray')
        size = sizes.get(atom, 100)
        ax.scatter(coord[0], coord[1], coord[2], 
                  c=color, s=size, edgecolors='black', linewidths=1, alpha=0.8)
        ax.text(coord[0], coord[1], coord[2], atom, fontsize=10)
    
    # 結合を描画（簡易版）
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            # 典型的な結合距離（1.0-1.8 Å）なら結合を描画
            if dist < 1.8:
                ax.plot([coords[i][0], coords[j][0]], 
                       [coords[i][1], coords[j][1]], 
                       [coords[i][2], coords[j][2]], 'k-', alpha=0.3)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)
    plt.tight_layout()
    return fig

def main():
    start_time = time.time()
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='構造最適化と振動数計算')
    parser.add_argument('--smiles', type=str, required=True, help='分子のSMILES')
    parser.add_argument('--basis', type=str, default='6-31+G**', help='基底関数')
    parser.add_argument('--charge', type=int, default=0, help='電荷')
    parser.add_argument('--spin', type=int, default=0, help='スピン多重度-1')
    parser.add_argument('--use-gpu', action='store_true', help='GPU加速を使用')
    args = parser.parse_args()
    
    print("="*60)
    print("構造最適化と振動数計算")
    print("="*60)
    print(f"SMILES: {args.smiles}")
    print(f"Method: B3LYP/{args.basis}")
    if args.use_gpu:
        print("GPU acceleration enabled.")
    
    with tqdm(total=5, desc="Overall Progress") as pbar:
        pbar.set_description("[1/5] 初期3D構造生成")
        atoms, init_coords = smiles_to_xyz(args.smiles)
        mol_rdkit = Chem.MolFromSmiles(args.smiles)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol_rdkit)
        print(f"分子式: {formula}, 原子数: {len(atoms)}")
        fig1 = visualize_molecule(atoms, init_coords, f"{formula} - 初期構造")
        plt.savefig(f"{formula}_initial.png", dpi=150, bbox_inches='tight')
        pbar.update(1)

        pbar.set_description("[2/5] PySCF分子オブジェクト作成")
        mol = create_mol(atoms, init_coords, args.basis, args.charge, args.spin)
        print(f"電子数: {mol.nelectron}, 基底関数数: {mol.nao}")
        pbar.update(1)

        pbar.set_description("[3/5] 構造最適化実行中")
        mf = dft.RKS(mol)
        mf.xc = 'B3LYP'
        if args.use_gpu:
            mf = mf.to_gpu()
        mf.kernel()
        e_init = mf.e_tot
        print(f"初期エネルギー: {e_init:.6f} Hartree")
        mol_opt = optimize(mf, maxsteps=50)
        mf_opt = dft.RKS(mol_opt)
        mf_opt.xc = 'B3LYP'
        if args.use_gpu:
            mf_opt = mf_opt.to_gpu()
        e_opt = mf_opt.kernel()
        print(f"最適化エネルギー: {e_opt:.6f} Hartree")
        print(f"エネルギー変化: {(e_opt - e_init)*627.509:.4f} kcal/mol")
        opt_coords = mol_opt.atom_coords() * 0.529177
        fig2 = visualize_molecule(atoms, opt_coords, f"{formula} - 最適化構造")
        plt.savefig(f"{formula}_optimized.png", dpi=150, bbox_inches='tight')
        rmsd = np.sqrt(np.mean(np.sum((init_coords - opt_coords)**2, axis=1)))
        print(f"構造変化RMSD: {rmsd:.4f} Å")
        pbar.update(1)

        pbar.set_description("[4/5] 振動数解析実行中")
        from pyscf import hessian
        h = hessian.rks.Hessian(mf_opt)
        if args.use_gpu:
            h = h.to_gpu()
        hess = h.kernel()
        freq_info = thermo.harmonic_analysis(mol_opt, hess)
        frequencies = freq_info['freq_wavenumber']
        n_imaginary = np.sum(frequencies < 0)
        print(f"虚振動数: {n_imaginary}個")
        if n_imaginary == 0:
            print("✅ 安定構造（極小点）")
        else:
            print("⚠️ 遷移状態または鞍点")
        real_freq = frequencies[frequencies >= 0]
        if len(real_freq) > 0:
            print(f"最低振動数: {real_freq[0]:.2f} cm⁻¹")
            print(f"最高振動数: {real_freq[-1]:.2f} cm⁻¹")
        pbar.update(1)

        pbar.set_description("[5/5] 熱力学的性質の計算")
        thermo_results = thermo.thermo(mf_opt, freq_info['freq_au'], 298.15, 101325.)
        zpe = thermo_results['ZPE']
        enthalpy = thermo_results['H_tot']
        gibbs = thermo_results['G_tot']
        entropy = thermo_results['S_tot']
        print(f"ゼロ点エネルギー: {zpe[0]*627.509:.3f} kcal/mol")
        print(f"エンタルピー: {enthalpy[0]:.6f} Hartree")
        print(f"ギブズ自由エネルギー: {gibbs[0]:.6f} Hartree")
        print(f"エントロピー: {entropy[0]*1000:.2f} cal/(mol·K)")
        pbar.update(1)
    
    # XYZファイル保存
    with open(f"{formula}_optimized.xyz", 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"Optimized structure E={e_opt:.6f} Hartree\n")
        for atom, coord in zip(atoms, opt_coords):
            f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
    
    print(f"\n最適化構造を {formula}_optimized.xyz に保存")
    
    # サマリー保存
    with open(f"{formula}_summary.txt", 'w') as f:
        f.write(f"SMILES: {args.smiles}\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Method: B3LYP/{args.basis}\n")
        f.write(f"Initial Energy: {e_init:.6f} Hartree\n")
        f.write(f"Optimized Energy: {e_opt:.6f} Hartree\n")
        f.write(f"Energy Change: {(e_opt - e_init)*627.509:.4f} kcal/mol\n")
        f.write(f"RMSD: {rmsd:.4f} Å\n")
        f.write(f"Imaginary Frequencies: {n_imaginary}\n")
        f.write(f"ZPE: {zpe[0]*627.509:.3f} kcal/mol\n")
        f.write(f"Gibbs Energy (298K): {gibbs[0]:.6f} Hartree\n")
    
    print(f"サマリーを {formula}_summary.txt に保存")
    
    # 両構造を並べて表示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': '3d'})
    
    # 初期構造
    for atom, coord in zip(atoms, init_coords):
        colors = {'H': 'white', 'C': 'gray', 'O': 'red', 'N': 'blue'}
        ax1.scatter(coord[0], coord[1], coord[2], 
                   c=colors.get(atom, 'gray'), s=100, edgecolors='black', alpha=0.8)
    ax1.set_title("初期構造")
    
    # 最適化構造
    for atom, coord in zip(atoms, opt_coords):
        colors = {'H': 'white', 'C': 'gray', 'O': 'red', 'N': 'blue'}
        ax2.scatter(coord[0], coord[1], coord[2],
                   c=colors.get(atom, 'gray'), s=100, edgecolors='black', alpha=0.8)
    ax2.set_title("最適化構造")
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
    
    plt.suptitle(f"{formula} 構造最適化前後の比較")
    plt.savefig(f"{formula}_comparison.png", dpi=150, bbox_inches='tight')
    print(f"比較図を {formula}_comparison.png に保存")
    
    print("\n" + "="*60)
    print("計算完了！")
    print("="*60)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"実行時間: {duration:.2f}秒")

    # matplotlibウィンドウを表示
    plt.show()

if __name__ == "__main__":
    main()
