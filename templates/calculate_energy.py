#!/usr/bin/env python3
"""
PySCF基本エネルギー計算と分子軌道解析
RTX 50シリーズGPU対応版 (Enhanced)

使用例:
python calculate_energy.py --smiles "CC(=O)O" --use-gpu
python calculate_energy.py --smiles "c1ccccc1" --method "B3LYP" --basis "6-31G*" --use-gpu
"""

import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pyscf import gto, scf, dft
import warnings
import sys
import os
import time
import re

# ログ出力用クラス（複数のストリームに出力）
class MultiWriter(object):
    def __init__(self, streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            try:
                stream.flush()
            except:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except:
                pass

# GPU利用可能性チェック
GPU4PYSCF_AVAILABLE = False
try:
    import cupy
    import gpu4pyscf
    from gpu4pyscf.dft import rks as gpu_rks
    from gpu4pyscf.scf import hf as gpu_hf
    GPU4PYSCF_AVAILABLE = True
    print("✅ gpu4pyscf is available - GPU acceleration enabled")
    # CuPyのバージョンとCUDAバージョンを確認
    try:
        print(f"   CuPy version: {cupy.__version__}")
        print(f"   CUDA version: {cupy.cuda.runtime.runtimeGetVersion()}")
    except:
        pass
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
    AllChem.MMFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    atoms = []
    coords = []
    
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(atom.GetSymbol())
        coords.append([pos.x, pos.y, pos.z])
    
    return atoms, np.array(coords)

def create_pyscf_mol(atoms, coords, basis='6-31G', charge=0, spin=0, output_stream=None):
    """PySCF分子オブジェクトを作成"""
    atom_str = ""
    for atom, coord in zip(atoms, coords):
        atom_str += f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}; "
    
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 4

    # PySCFの出力ストリームを設定
    if output_stream is not None:
        mol.output = None
        mol.stdout = output_stream

    try:
        mol.build()
    except Exception as e:
        # BasisNotFoundError (pyscf.lib.exceptions.BasisNotFoundError) などを捕捉
        if "Basis not found" in str(e) or "Basis data not found" in str(e):
            print(f"\n⚠️ Basis set '{basis}' not found for some atoms.")
            print("   Attempting fallback to 'def2-SVP' (supports most elements)...")
            mol.basis = 'def2-SVP'
            mol.build()
            print("   ✅ Fallback successful using 'def2-SVP'")
        else:
            raise e
    
    return mol

def perform_calculation(mol, method='HF', use_gpu=False):
    """エネルギー計算を実行（堅牢なGPU/CPUフォールバック機能付き）"""
    
    mf = None
    energy = None

    # GPU計算試行
    if use_gpu and GPU4PYSCF_AVAILABLE:
        try:
            print(f"🚀 Attempting GPU calculation for {method}...")
            
            # MP2の場合
            if method == 'MP2':
                # Step 1: HF on GPU
                print("   Step 1: RHF calculation (GPU)...")
                mf_hf = gpu_hf.RHF(mol)
                # 初期推定の改善（opt-freq.pyと同様のロジック）
                try:
                    mf_hf.init_guess = 'atom'
                    mf_hf = mf_hf.to_gpu()
                    mf_hf.kernel()
                except Exception as e_guess:
                     print(f"   Direct GPU RHF failed ({e_guess}), trying hybrid approach...")
                     # CPU Guess -> GPU
                     mf_cpu = scf.RHF(mol)
                     mf_cpu.max_cycle = 5
                     mf_cpu.kernel()
                     dm = mf_cpu.make_rdm1()
                     mf_hf = gpu_hf.RHF(mol).to_gpu()
                     mf_hf.kernel(dm0=dm)

                # Step 2: MP2 on GPU
                print("   Step 2: MP2 calculation (GPU)...")
                from gpu4pyscf import mp
                mp2 = mp.MP2(mf_hf)
                mp2.kernel()
                return mf_hf, mp2.e_tot

            # HF/DFTの場合
            else:
                if method == 'HF':
                    mf = gpu_hf.RHF(mol)
                else: # DFT
                    mf = gpu_rks.RKS(mol)
                    mf.xc = method

                # Direct GPU try
                try:
                    mf.init_guess = 'atom'
                    mf = mf.to_gpu()
                    energy = mf.kernel()
                    return mf, energy
                except Exception as e_direct:
                     print(f"   Direct GPU calculation failed ({e_direct}), trying hybrid approach...")
                     # Hybrid: CPU Guess -> GPU
                     if method == 'HF':
                         mf_cpu = scf.RHF(mol)
                     else:
                         mf_cpu = dft.RKS(mol)
                         mf_cpu.xc = method

                     mf_cpu.max_cycle = 5
                     mf_cpu.kernel()
                     dm = mf_cpu.make_rdm1()

                     if method == 'HF':
                         mf = gpu_hf.RHF(mol)
                     else:
                         mf = gpu_rks.RKS(mol)
                         mf.xc = method

                     mf = mf.to_gpu()
                     energy = mf.kernel(dm0=dm)
                     return mf, energy

        except Exception as e:
            print(f"⚠️ GPU calculation failed: {e}")
            print("   Falling back to CPU...")
    
    # CPU計算（フォールバックまたは最初からCPU指定）
    if use_gpu and not GPU4PYSCF_AVAILABLE:
        print("⚠️ GPU requested but gpu4pyscf not available.")
    print("💻 Using CPU calculation...")

    if method == 'MP2':
        print("   Step 1: RHF calculation (CPU)...")
        mf = scf.RHF(mol)
        mf.kernel()
        print("   Step 2: MP2 calculation (CPU)...")
        from pyscf import mp
        mp2 = mp.MP2(mf)
        mp2.kernel()
        return mf, mp2.e_tot

    elif method == 'HF':
        mf = scf.RHF(mol)
    else: # DFT
        mf = dft.RKS(mol)
        mf.xc = method
    
    energy = mf.kernel()
    return mf, energy

def analyze_orbitals(mf, mol):
    """分子軌道解析"""
    # GPUオブジェクトの場合、CPUへ戻す必要がある場合があるが
    # mo_energy等は通常numpy配列またはcupy配列としてアクセス可能
    # 安全のためCPUへ持ってくる

    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    
    # CuPy配列ならNumPyに変換
    if hasattr(mo_energy, 'get'):
        mo_energy = mo_energy.get()
    if hasattr(mo_occ, 'get'):
        mo_occ = mo_occ.get()

    # HOMO/LUMO
    # 占有軌道の最後のインデックスを探す
    try:
        homo_idx = np.where(mo_occ > 0)[0][-1]
        lumo_idx = homo_idx + 1

        homo_energy = mo_energy[homo_idx]
        lumo_energy = mo_energy[lumo_idx] if lumo_idx < len(mo_energy) else None
        gap = lumo_energy - homo_energy if lumo_energy else None

        return {
            'homo': homo_energy,
            'lumo': lumo_energy,
            'gap': gap,
            'homo_idx': homo_idx,
            'lumo_idx': lumo_idx
        }
    except Exception as e:
        print(f"Orbital analysis failed: {e}")
        return {'homo': 0, 'lumo': 0, 'gap': 0, 'homo_idx': 0, 'lumo_idx': 0}

def calculate_dipole(mf):
    """双極子モーメント計算"""
    # GPUオブジェクトの場合、CPUへ変換して計算するのが安全
    if hasattr(mf, 'to_cpu'):
        mf_cpu = mf.to_cpu()
    else:
        mf_cpu = mf

    dm = mf_cpu.make_rdm1()
    dipole = mf_cpu.dip_moment(mf_cpu.mol, dm, unit='Debye')
    return dipole

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description='PySCF基本エネルギー計算')
    parser.add_argument('--smiles', type=str, required=True, 
                       help='計算対象分子のSMILES')
    parser.add_argument('--method', type=str, default='HF',
                       choices=['HF', 'B3LYP', 'PBE', 'M06-2X', 'MP2'],
                       help='計算手法 (default: HF)')
    parser.add_argument('--basis', type=str, default='6-31G',
                       help='基底関数 (default: 6-31G)')
    parser.add_argument('--charge', type=int, default=0,
                       help='分子の電荷 (default: 0)')
    parser.add_argument('--spin', type=int, default=0,
                       help='スピン多重度-1 (default: 0)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='GPU加速を使用')
    
    args = parser.parse_args()
    
    # ファイル名用のSMILESサニタイズ
    safe_smiles = re.sub(r'[\\/:\*\?"<>\|]', '_', args.smiles)

    # ログファイルの設定
    # 命名規則: {SMILES}_{script}_{method}_{basis}_{type}.txt
    base_name = f"{safe_smiles}_calculate_energy_{args.method}_{args.basis}"
    short_log_name = f"{base_name}_short_report.txt"
    full_log_name = f"{base_name}_log_report.txt"

    f_short = open(short_log_name, "w")
    f_full = open(full_log_name, "w")

    original_stdout = sys.stdout
    sys.stdout = MultiWriter([original_stdout, f_short, f_full])

    # PySCFの詳細ログ用Writer (Terminal + Full log)
    pyscf_writer = MultiWriter([original_stdout, f_full])

    print("=" * 60)
    print("PySCF エネルギー計算")
    print("=" * 60)
    print(f"SMILES: {args.smiles}")
    print(f"手法: {args.method}/{args.basis}")
    print(f"要約ログ: {short_log_name}")
    print(f"詳細ログ: {full_log_name}")
    
    try:
        # 分子情報取得
        mol_rdkit = Chem.MolFromSmiles(args.smiles)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol_rdkit)
        mw = Descriptors.MolWt(mol_rdkit)
        print(f"分子式: {formula}")
        print(f"分子量: {mw:.2f}")

        # 3D構造生成
        print("\n[1] 3D構造生成...")
        atoms, coords = smiles_to_xyz(args.smiles)

        # PySCF分子作成
        print("[2] PySCF分子オブジェクト作成...")
        mol = create_pyscf_mol(atoms, coords, args.basis, args.charge, args.spin, output_stream=pyscf_writer)
        print(f"原子数: {mol.natm}")
        print(f"電子数: {mol.nelectron}")
        print(f"基底関数数: {mol.nao}")

        # エネルギー計算
        print(f"\n[3] {args.method}計算実行中...")
        mf, energy = perform_calculation(mol, args.method, args.use_gpu)
        print(f"✅ 全エネルギー: {energy:.6f} Hartree")
        print(f"   = {energy * 27.2114:.4f} eV")
        print(f"   = {energy * 627.509:.2f} kcal/mol")

        # 軌道解析
        print("\n[4] 分子軌道解析...")
        orbital_info = analyze_orbitals(mf, mol)
        print(f"HOMO エネルギー: {orbital_info['homo']:.4f} Hartree")
        if orbital_info['lumo']:
            print(f"LUMO エネルギー: {orbital_info['lumo']:.4f} Hartree")
            print(f"HOMO-LUMO ギャップ: {orbital_info['gap']:.4f} Hartree")
            print(f"                   = {orbital_info['gap']*27.2114:.2f} eV")

        # 双極子モーメント
        print("\n[5] 分子特性...")
        try:
            dipole = calculate_dipole(mf)
            dipole_mag = np.linalg.norm(dipole)
            print(f"双極子モーメント: {dipole_mag:.4f} Debye")
            print(f"  成分 (x,y,z): [{dipole[0]:.3f}, {dipole[1]:.3f}, {dipole[2]:.3f}]")
        except Exception as e:
            print(f"双極子モーメント計算エラー: {e}")
            dipole_mag = 0.0

        # 結果サマリー
        print("\n" + "=" * 60)
        print("計算完了！")
        print("=" * 60)

        end_time = time.time()
        print(f"実行時間: {end_time - start_time:.2f}秒")

        # 結果をファイルに保存 (従来の要約ファイル)
        output_file = f"{formula}_{args.method}_{args.basis}.txt"
        with open(output_file, 'w') as f:
            f.write(f"SMILES: {args.smiles}\n")
            f.write(f"Formula: {formula}\n")
            f.write(f"Method: {args.method}/{args.basis}\n")
            f.write(f"Total Energy: {energy:.6f} Hartree\n")
            f.write(f"HOMO: {orbital_info['homo']:.4f} Hartree\n")
            if orbital_info['lumo']:
                f.write(f"LUMO: {orbital_info['lumo']:.4f} Hartree\n")
                f.write(f"Gap: {orbital_info['gap']*27.2114:.2f} eV\n")
            f.write(f"Dipole: {dipole_mag:.4f} Debye\n")

        print(f"結果サマリーを {output_file} に保存しました")

    finally:
        f_short.close()
        f_full.close()
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()
