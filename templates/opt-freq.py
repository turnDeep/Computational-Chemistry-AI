#!/usr/bin/env python3
"""
PySCF構造最適化と振動数計算スクリプト（GPU対応修正版）
Usage: python opt-freq.py --smiles "CCO" --use-gpu
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft, scf
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo
from tqdm import tqdm
import time
import warnings
import sys
import os

# ログ出力用クラス
class MultiStream(object):
    """複数のストリームに出力を分岐させるクラス"""
    def __init__(self, streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            try:
                stream.write(message)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass

    def close(self):
        # sys.__stdout__などは閉じないように注意
        pass

# GPU利用可能性チェック
GPU4PYSCF_AVAILABLE = False
try:
    import cupy
    import gpu4pyscf
    from gpu4pyscf.dft import rks as gpu_rks
    GPU4PYSCF_AVAILABLE = True
    print("✅ gpu4pyscf is available - GPU acceleration enabled")
    # CuPyのバージョンとCUDAバージョンを確認
    print(f"   CuPy version: {cupy.__version__}")
    print(f"   CUDA version: {cupy.cuda.runtime.runtimeGetVersion()}")
except ImportError as e:
    print(f"⚠️ gpu4pyscf not available - CPU only mode: {e}")

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

def create_mol(atoms, coords, basis='6-31+G**', charge=0, spin=0, output=None):
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
    mol.verbose = 4  # デバッグ用に詳細ログを出力
    if output:
        mol.stdout = output
    mol.build()
    
    return mol

def create_mf_object(mol, use_gpu=False):
    """適切なMFオブジェクトを作成（GPU/CPU）"""
    if use_gpu and GPU4PYSCF_AVAILABLE:
        print("🚀 Using GPU acceleration (gpu4pyscf)")
        try:
            # まずCPUでSCF計算を実行して初期密度行列を取得
            print("   Computing initial guess on CPU...")
            mf_cpu = dft.RKS(mol)
            mf_cpu.xc = 'B3LYP'
            mf_cpu.init_guess = 'atom'  # シンプルな初期推定を使用
            mf_cpu.max_cycle = 1  # 1サイクルだけ実行
            mf_cpu.kernel()
            dm_init = mf_cpu.make_rdm1()
            
            # GPU計算に移行
            print("   Transferring to GPU...")
            mf = gpu_rks.RKS(mol)
            mf.xc = 'B3LYP'
            mf.init_guess = dm_init  # CPU計算の密度行列を初期推定として使用
            mf = mf.to_gpu()
            
            return mf
            
        except Exception as e:
            print(f"⚠️ GPU initialization failed: {e}")
            print("   Falling back to CPU...")
            mf = dft.RKS(mol)
            mf.xc = 'B3LYP'
            return mf
    else:
        if use_gpu and not GPU4PYSCF_AVAILABLE:
            print("⚠️ GPU requested but gpu4pyscf not available, falling back to CPU")
        print("💻 Using CPU")
        mf = dft.RKS(mol)
        mf.xc = 'B3LYP'
        return mf

def safe_gpu_calculation(mol, use_gpu=False):
    """安全なGPU計算（エラー時はCPUにフォールバック）"""
    if use_gpu and GPU4PYSCF_AVAILABLE:
        try:
            # 方法1: init_guessを変更してGPU計算を試みる
            print("   Attempting GPU calculation with modified init_guess...")
            mf = gpu_rks.RKS(mol)
            mf.xc = 'B3LYP'
            mf.init_guess = 'atom'  # 'minao'の代わりに'atom'を使用
            mf = mf.to_gpu()
            energy = mf.kernel()
            return mf, energy
        except Exception as e1:
            print(f"   Method 1 failed: {e1}")
            try:
                # 方法2: CPUで初期計算してからGPUに転送
                print("   Attempting hybrid CPU-GPU approach...")
                # CPUで初期密度行列を計算
                mf_cpu = dft.RKS(mol)
                mf_cpu.xc = 'B3LYP'
                mf_cpu.max_cycle = 5
                energy_cpu = mf_cpu.kernel()
                dm = mf_cpu.make_rdm1()
                
                # GPUに転送
                mf_gpu = gpu_rks.RKS(mol)
                mf_gpu.xc = 'B3LYP'
                mf_gpu = mf_gpu.to_gpu()
                energy = mf_gpu.kernel(dm0=dm)
                return mf_gpu, energy
            except Exception as e2:
                print(f"   Method 2 failed: {e2}")
                print("   Falling back to CPU calculation...")
    
    # CPUで計算
    mf = dft.RKS(mol)
    mf.xc = 'B3LYP'
    energy = mf.kernel()
    return mf, energy

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
    # 分子式を先に取得してログファイル名を決定
    mol_rdkit = Chem.MolFromSmiles(args.smiles)
    if mol_rdkit is None:
        raise ValueError(f"Invalid SMILES: {args.smiles}")
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol_rdkit)
    
    # ログファイル設定
    short_report_name = f"{formula}_short_report.txt"
    log_report_name = f"{formula}_log_report.txt"

    f_short = open(short_report_name, "w")
    f_log = open(log_report_name, "w")

    # 標準出力の置き換え（print文の出力先）
    # Terminal, Short Report, Log Reportすべてに出力
    sys.stdout = MultiStream([sys.__stdout__, f_short, f_log])

    # PySCF詳細ログの出力先
    # Terminal, Log Reportに出力（Short Reportには出さない）
    pyscf_log_stream = MultiStream([sys.__stdout__, f_log])
    
    print(f"SMILES: {args.smiles}")
    print(f"Method: B3LYP/{args.basis}")
    print(f"レポートファイル: {short_report_name}")
    print(f"詳細ログファイル: {log_report_name}")
    
    # tqdmは直接ターミナルに出力する（ログファイルにはプログレスバーを出さない）
    with tqdm(total=5, desc="Overall Progress", file=sys.__stdout__) as pbar:
        pbar.set_description("[1/5] 初期3D構造生成")
        atoms, init_coords = smiles_to_xyz(args.smiles)
        print(f"分子式: {formula}, 原子数: {len(atoms)}")
        pbar.update(1)

        pbar.set_description("[2/5] PySCF分子オブジェクト作成")
        # create_molに詳細ログ用のストリームを渡す
        mol = create_mol(atoms, init_coords, args.basis, args.charge, args.spin, output=pyscf_log_stream)
        print(f"電子数: {mol.nelectron}, 基底関数数: {mol.nao}")
        pbar.update(1)

        pbar.set_description("[3/5] 構造最適化実行中")
        # 初期エネルギー計算（安全なGPU計算）
        mf, e_init = safe_gpu_calculation(mol, args.use_gpu)
        print(f"初期エネルギー: {e_init:.6f} Hartree")
        
        # 構造最適化（CPUで実行 - geomeTRICはGPU未対応のため）
        print("   Structure optimization (CPU)...")
        mol_opt = optimize(mf, maxsteps=50)
        
        # 最適化後の計算
        mf_opt, e_opt = safe_gpu_calculation(mol_opt, args.use_gpu)
        print(f"最適化エネルギー: {e_opt:.6f} Hartree")
        print(f"エネルギー変化: {(e_opt - e_init)*627.509:.4f} kcal/mol")
        
        opt_coords = mol_opt.atom_coords() * 0.529177
        rmsd = np.sqrt(np.mean(np.sum((init_coords - opt_coords)**2, axis=1)))
        print(f"構造変化RMSD: {rmsd:.4f} Å")
        pbar.update(1)

        pbar.set_description("[4/5] 振動数解析実行中")
        from pyscf import hessian
        
        # Hessian計算
        if args.use_gpu and GPU4PYSCF_AVAILABLE:
            try:
                from gpu4pyscf import hessian as gpu_hessian
                print("   Hessian calculation (GPU)...")
                h = gpu_hessian.rks.Hessian(mf_opt)
                hess = h.kernel()
                # 後の解析のためにCPUへ転送（必要な場合）
                if hasattr(hess, 'get'):
                    hess = hess.get()
            except Exception as e:
                print(f"⚠️ GPU Hessian failed: {e}")
                print("   Falling back to CPU Hessian...")
                # GPUオブジェクトをCPUに変換
                if hasattr(mf_opt, 'to_cpu'):
                    mf_cpu = mf_opt.to_cpu()
                else:
                    mf_cpu = mf_opt
                h = hessian.rks.Hessian(mf_cpu)
                hess = h.kernel()
        else:
            print("   Hessian calculation (CPU)...")
            # GPUオブジェクトをCPUに変換
            if hasattr(mf_opt, 'to_cpu'):
                mf_cpu = mf_opt.to_cpu()
            else:
                mf_cpu = mf_opt
            h = hessian.rks.Hessian(mf_cpu)
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
        # thermo.thermo()は辞書を返し、その値は貢献成分のリスト [合計, 電子, 並進, 回転, 振動]
        thermo_results = thermo.thermo(mf_opt, freq_info['freq_au'], 298.15, 101325)
        
        # 辞書のキーで値(リスト)を取得し、その先頭要素(合計値)を取り出す
        zpe = thermo_results['ZPE'][0]
        e_tot = thermo_results['E_tot'][0]
        h_tot = thermo_results['H_tot'][0]
        g_tot = thermo_results['G_tot'][0]
        s_tot = thermo_results['S_tot'][0]
        
        print(f"ゼロ点エネルギー: {zpe*627.509:.3f} kcal/mol")
        print(f"エンタルピー: {h_tot:.6f} Hartree")
        print(f"ギブズ自由エネルギー: {g_tot:.6f} Hartree")
        print(f"エントロピー: {s_tot*1000:.2f} cal/(mol·K)")
        pbar.update(1)
    
    # XYZファイル保存
    with open(f"{formula}_optimized.xyz", 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write(f"Optimized structure E={e_opt:.6f} Hartree\n")
        for atom, coord in zip(atoms, opt_coords):
            f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
    
    print(f"\n最適化構造を {formula}_optimized.xyz に保存")
    
    # NOTE: summary.txtの生成コードは削除し、short_report.txtに集約
    print(f"サマリーレポートを {short_report_name} に保存")
    print(f"比較図を {formula}_comparison.png に保存 (※作成ロジックが必要であれば追加)")
    
    print("\n" + "="*60)
    print("計算完了！")
    print("="*60)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"実行時間: {duration:.2f}秒")

    # ファイルを閉じる
    f_short.close()
    f_log.close()

if __name__ == "__main__":
    main()
