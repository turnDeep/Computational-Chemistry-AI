#!/usr/bin/env python3
"""
PySCF構造最適化と振動数解析
geomeTRICを使用した高精度構造最適化

使用例:
python optimize_geometry.py --smiles "CCO"  # エタノール
python optimize_geometry.py --smiles "CC(=O)O" --freq  # 酢酸の振動数計算も実行
"""

import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pyscf import gto, scf, dft, hessian
from pyscf.geomopt.geometric_solver import optimize
import torch
import warnings
warnings.filterwarnings('ignore')

def smiles_to_xyz(smiles):
    """SMILESから初期3D座標を生成"""
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

def optimize_geometry(mol, method='B3LYP', use_gpu=False, max_steps=50):
    """構造最適化を実行"""
    
    print(f"初期構造エネルギー計算中...")
    
    # GPU利用可能性チェック
    if use_gpu and torch.cuda.is_available():
        try:
            import gpu4pyscf
            print(f"✅ GPU使用: {torch.cuda.get_device_name(0)}")
            
            if method == 'HF':
                mf = gpu4pyscf.scf.RHF(mol).to_gpu()
            else:
                mf = gpu4pyscf.dft.RKS(mol).to_gpu()
                mf.xc = method
        except ImportError:
            print("⚠️ gpu4pyscf未インストール、CPUを使用")
            use_gpu = False
            if method == 'HF':
                mf = scf.RHF(mol)
            else:
                mf = dft.RKS(mol)
                mf.xc = method
    else:
        if method == 'HF':
            mf = scf.RHF(mol)
        else:
            mf = dft.RKS(mol)
            mf.xc = method
    
    # 初期エネルギー
    e_init = mf.kernel()
    print(f"初期エネルギー: {e_init:.6f} Hartree")
    
    # 構造最適化
    print("\n構造最適化実行中...")
    print("Step    Energy (Hartree)    RMS Force")
    print("-" * 45)
    
    # geomeTRICによる最適化
    mol_opt = optimize(mf, maxsteps=max_steps)
    
    # 最適化後の計算
    if method == 'HF':
        mf_opt = scf.RHF(mol_opt)
    else:
        mf_opt = dft.RKS(mol_opt)
        mf_opt.xc = method
    
    e_opt = mf_opt.kernel()
    
    return mol_opt, mf_opt, e_init, e_opt

def calculate_frequencies(mf):
    """振動数解析"""
    print("\n振動数解析実行中...")
    
    # Hessian計算
    if isinstance(mf, (scf.hf.RHF, scf.uhf.UHF)):
        h = hessian.RHF(mf)
    else:
        h = hessian.RKS(mf)
    
    # Hessian行列計算
    hess = h.kernel()
    
    # 振動解析
    from pyscf.hessian import thermo
    freq_info = thermo.harmonic_analysis(mf.mol, hess)
    
    # 周波数をcm^-1に変換
    frequencies = freq_info['freq_wavenumber']
    
    # 虚振動チェック
    n_imaginary = np.sum(frequencies < 0)
    
    return frequencies, n_imaginary, freq_info

def calculate_thermodynamics(freq_info, T=298.15):
    """熱力学的特性を計算"""
    from pyscf.hessian import thermo
    
    results = thermo.thermo(
        freq_info['freq_au'],
        T,
        pressure=101325  # 1 atm in Pa
    )
    
    return {
        'temperature': T,
        'ZPE': results[0],  # Zero-point energy
        'E_tot': results[1],  # Total thermal energy
        'H': results[2],     # Enthalpy
        'G': results[3],     # Gibbs free energy
        'S': results[4],     # Entropy
    }

def save_xyz_file(atoms, coords, filename):
    """XYZファイルを保存"""
    with open(filename, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write("Optimized structure\n")
        for atom, coord in zip(atoms, coords):
            f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

def main():
    parser = argparse.ArgumentParser(description='PySCF構造最適化')
    parser.add_argument('--smiles', type=str, required=True,
                       help='計算対象分子のSMILES')
    parser.add_argument('--method', type=str, default='B3LYP',
                       choices=['HF', 'B3LYP', 'PBE', 'M06-2X'],
                       help='計算手法 (default: B3LYP)')
    parser.add_argument('--basis', type=str, default='6-31G*',
                       help='基底関数 (default: 6-31G*)')
    parser.add_argument('--charge', type=int, default=0,
                       help='分子の電荷 (default: 0)')
    parser.add_argument('--spin', type=int, default=0,
                       help='スピン多重度-1 (default: 0)')
    parser.add_argument('--freq', action='store_true',
                       help='振動数解析を実行')
    parser.add_argument('--use-gpu', action='store_true',
                       help='GPU加速を使用')
    parser.add_argument('--max-steps', type=int, default=50,
                       help='最大最適化ステップ数 (default: 50)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PySCF 構造最適化")
    print("=" * 60)
    print(f"SMILES: {args.smiles}")
    print(f"手法: {args.method}/{args.basis}")
    
    # 分子情報取得
    mol_rdkit = Chem.MolFromSmiles(args.smiles)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol_rdkit)
    mw = Descriptors.MolWt(mol_rdkit)
    print(f"分子式: {formula}")
    print(f"分子量: {mw:.2f} g/mol")
    
    # 初期構造生成
    print("\n[1] 初期3D構造生成...")
    atoms, coords = smiles_to_xyz(args.smiles)
    print(f"原子数: {len(atoms)}")
    
    # PySCF分子作成
    print("\n[2] PySCF分子オブジェクト作成...")
    mol = create_pyscf_mol(atoms, coords, args.basis, args.charge, args.spin)
    print(f"電子数: {mol.nelectron}")
    print(f"基底関数数: {mol.nao}")
    
    # 構造最適化
    print(f"\n[3] 構造最適化 ({args.method}/{args.basis})...")
    mol_opt, mf_opt, e_init, e_opt = optimize_geometry(
        mol, args.method, args.use_gpu, args.max_steps
    )
    
    print("\n最適化完了！")
    print(f"初期エネルギー: {e_init:.6f} Hartree")
    print(f"最適化エネルギー: {e_opt:.6f} Hartree")
    print(f"エネルギー変化: {(e_opt - e_init)*627.509:.4f} kcal/mol")
    
    # 最適化構造を取得
    opt_coords = mol_opt.atom_coords() * 0.529177  # Bohr to Angstrom
    
    # 構造変化を計算
    rmsd = np.sqrt(np.mean(np.sum((coords - opt_coords)**2, axis=1)))
    print(f"構造RMSD: {rmsd:.4f} Å")
    
    # 振動数解析
    if args.freq:
        print("\n[4] 振動数解析...")
        frequencies, n_imaginary, freq_info = calculate_frequencies(mf_opt)
        
        print(f"虚振動数: {n_imaginary}個")
        if n_imaginary == 0:
            print("✅ 安定構造（極小点）です")
        else:
            print("⚠️ 遷移状態または鞍点の可能性があります")
        
        # 振動数表示（主要なもののみ）
        print("\n振動数 (cm⁻¹):")
        sorted_freq = np.sort(frequencies)
        print("最低周波数（6個）:")
        for i, freq in enumerate(sorted_freq[:6]):
            print(f"  {i+1:2d}: {freq:8.2f} cm⁻¹")
        
        print("\n最高周波数（6個）:")
        for i, freq in enumerate(sorted_freq[-6:]):
            print(f"  {len(sorted_freq)-5+i:2d}: {freq:8.2f} cm⁻¹")
        
        # 熱力学的特性
        print("\n[5] 熱力学的特性 (298.15 K, 1 atm)...")
        thermo_data = calculate_thermodynamics(freq_info)
        
        print(f"ゼロ点エネルギー (ZPE): {thermo_data['ZPE']*627.509:.3f} kcal/mol")
        print(f"熱エネルギー補正: {thermo_data['E_tot']*627.509:.3f} kcal/mol")
        print(f"エンタルピー (H): {thermo_data['H']*627.509:.3f} kcal/mol")
        print(f"ギブズ自由エネルギー (G): {thermo_data['G']*627.509:.3f} kcal/mol")
        print(f"エントロピー (S): {thermo_data['S']*1000:.2f} cal/(mol·K)")
    
    # 結果を保存
    xyz_file = f"{formula}_optimized.xyz"
    save_xyz_file(atoms, opt_coords, xyz_file)
    print(f"\n最適化構造を {xyz_file} に保存しました")
    
    # サマリーファイル保存
    summary_file = f"{formula}_optimization_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"SMILES: {args.smiles}\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Method: {args.method}/{args.basis}\n")
        f.write(f"Initial Energy: {e_init:.6f} Hartree\n")
        f.write(f"Optimized Energy: {e_opt:.6f} Hartree\n")
        f.write(f"Energy Change: {(e_opt - e_init)*627.509:.4f} kcal/mol\n")
        f.write(f"Structure RMSD: {rmsd:.4f} Å\n")
        
        if args.freq:
            f.write(f"\nVibrational Analysis:\n")
            f.write(f"Imaginary Frequencies: {n_imaginary}\n")
            f.write(f"ZPE: {thermo_data['ZPE']*627.509:.3f} kcal/mol\n")
            f.write(f"Gibbs Free Energy (298K): {thermo_data['G']*627.509:.3f} kcal/mol\n")
    
    print(f"サマリーを {summary_file} に保存しました")
    
    print("\n" + "=" * 60)
    print("計算完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()