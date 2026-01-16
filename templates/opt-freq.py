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
try:
    from pyscf.prop import infrared
    HAS_INFRARED = True
except ImportError:
    HAS_INFRARED = False

from tqdm import tqdm
import time
import warnings
import sys
import os
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

def create_mol(atoms, coords, basis='6-31+G**', charge=0, spin=0, output_stream=None):
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

    # PySCFの出力ストリームを設定（ファイル名ではなくストリームオブジェクトを使用）
    if output_stream is not None:
        mol.output = None  # Noneにすることでstdout属性が使われる
        mol.stdout = output_stream

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

def numerical_ir_intensities(mol, mf, modes):
    """
    数値微分によるIR強度計算 (pyscf.prop.infraredがない場合の代替)

    Args:
        mol: PySCF Mole object
        mf: PySCF SCF object (converged)
        modes: Normal modes (n_modes, n_atoms, 3)
    Returns:
        intensities: IR intensities in km/mol
    """
    print("   ⚠️ pyscf.prop.infrared not found. Using numerical differentiation for IR intensities.")
    print("      This involves 6N SCF calculations and may be slow.")

    n_atom = mol.natm

    # 基準の双極子モーメント
    mu0 = mf.dip_moment(unit='au') # (3,)

    # 双極子微分の計算 (dmu/dx) -> (3, 3*natm)
    # mu_x, mu_y, mu_z for each coordinate displacement
    dip_grad = np.zeros((3, n_atom, 3)) # (dipole_comp, atom, coord)

    delta = 0.001 # Displacement size in Bohr

    # 現在の座標 (Angstrom)
    coords_orig = mol.atom_coords(unit='Bohr')

    # リスタート用に密度行列を保存
    dm0 = mf.make_rdm1()

    # 各原子・各座標についてループ
    cnt = 0
    total_steps = n_atom * 3

    print(f"   Calculating dipole derivatives ({total_steps} steps)...")

    # tqdmを使いたいが、MultiWriterと競合する可能性があるのでシンプルなprintで

    for i in range(n_atom):
        for j in range(3): # x, y, z
            # +変位
            coords_plus = coords_orig.copy()
            coords_plus[i, j] += delta

            mol_plus = mol.copy()
            mol_plus.set_geom_(coords_plus, unit='Bohr')
            # 密度行列を初期値にして計算高速化
            if isinstance(mf, dft.uks.UKS):
                mf_plus = dft.UKS(mol_plus)
                mf_plus.xc = mf.xc
            elif isinstance(mf, dft.rks.RKS):
                mf_plus = dft.RKS(mol_plus)
                mf_plus.xc = mf.xc
            elif isinstance(mf, scf.uhf.UHF):
                mf_plus = scf.UHF(mol_plus)
            else:
                mf_plus = scf.RHF(mol_plus)

            mf_plus.verbose = 0
            mf_plus.kernel(dm0=dm0)
            mu_plus = mf_plus.dip_moment(unit='au')

            # -変位
            coords_minus = coords_orig.copy()
            coords_minus[i, j] -= delta

            mol_minus = mol.copy()
            mol_minus.set_geom_(coords_minus, unit='Bohr')

            if isinstance(mf, dft.uks.UKS):
                mf_minus = dft.UKS(mol_minus)
                mf_minus.xc = mf.xc
            elif isinstance(mf, dft.rks.RKS):
                mf_minus = dft.RKS(mol_minus)
                mf_minus.xc = mf.xc
            elif isinstance(mf, scf.uhf.UHF):
                mf_minus = scf.UHF(mol_minus)
            else:
                mf_minus = scf.RHF(mol_minus)

            mf_minus.verbose = 0
            mf_minus.kernel(dm0=dm0)
            mu_minus = mf_minus.dip_moment(unit='au')

            # 中心差分 (dmu / dR)
            deriv = (mu_plus - mu_minus) / (2.0 * delta)
            dip_grad[:, i, j] = deriv

            cnt += 1
            if cnt % 10 == 0:
                print(f"      Step {cnt}/{total_steps}")

    # 双極子微分行列を (3, 3N) に変形
    # dipole_deriv: (3, 3N) where 3N are nuclear coordinates
    dipole_deriv = dip_grad.reshape(3, -1) # (3, 3N)

    # ノーマルモードを (3N, n_modes) に変形
    # modes input is (n_modes, n_atoms, 3)
    # pyscf returns modes in mass-weighted coordinates? No, usually normalized displacements.
    # We need modes in shape (3N, n_modes)
    n_modes = modes.shape[0]

    # 振動モードごとの双極子変化 dmu/dQ = sum(dmu/dR * dR/dQ)
    # dR/dQ is the normal mode vector

    intensities = []

    for m in range(n_modes):
        mode_vec = modes[m].reshape(-1) # (3N,)

        # dmu_dQ = dipole_deriv (3, 3N) . mode_vec (3N, 1)
        dmu_dQ = np.dot(dipole_deriv, mode_vec) # (3,)

        # Intensity formula: I = (N_A * pi / 3c^2) * |dmu/dQ|^2
        # In km/mol units.
        # PySCF uses: 42.2561 * |dmu/dQ|^2  (if dmu/dQ is in au) -> km/mol
        # Check units:
        # dipole in au (Debye * ..), coordinates in Bohr.
        # factor ~ 42.2561

        val = np.sum(dmu_dQ**2)
        intens = val * 42.2561
        intensities.append(intens)

    return np.array(intensities)

def plot_ir_spectrum(frequencies, intensities, formula, save_file):
    """
    IRスペクトルをプロットして保存する

    Args:
        frequencies (array): 振動数 (cm^-1)
        intensities (array): IR強度 (km/mol)
        formula (str): 分子式
        save_file (str): 保存ファイルパス
    """
    # ガウシアンブロードニングの設定
    # x軸: 4000 cm^-1 (左) -> 400 cm^-1 (右)
    x = np.linspace(400, 4000, 2000)
    y = np.zeros_like(x)
    sigma = 20.0  # 半値幅パラメータ (cm^-1)

    # スペクトル生成 (強度の重ね合わせ)
    for freq, intensity in zip(frequencies, intensities):
        if freq > 0 and intensity > 0.1:  # 正の振動数かつ一定強度以上
            # ガウス関数: I * exp(-0.5 * ((x-freq)/sigma)^2)
            gaussian = intensity * np.exp(-0.5 * ((x - freq) / sigma) ** 2)
            y += gaussian

    # プロット作成
    plt.figure(figsize=(10, 6))

    # 透過率(Transmittance)風に表示 (最大強度を100%吸収として、100% - Absorbance)
    # 実際の透過率は濃度や光路長に依存するため、ここでは定性的な「吸収の逆」としてプロット
    if np.max(y) > 0:
        # 正規化して透過率(%)に変換 (100% -> 0% の範囲に収める)
        # y_norm = y / np.max(y) * 80 # 最大吸収を80%とする
        # transmittance = 100 - y_norm

        # 吸光度(Absorbance)としてプロットする方が物理的には正確だが、化学者は透過率スペクトルに見慣れている
        # ここでは単純に強度(Absorbance)をプロットし、y軸を反転させる手法をとる（透過率風に見える）
        plt.plot(x, y, color='blue', linewidth=1.5)
        plt.fill_between(x, y, color='blue', alpha=0.1)
        plt.ylabel('Simulated Absorption Intensity (arb. units)')
    else:
        plt.plot(x, y, color='blue')
        plt.ylabel('Intensity')

    plt.title(f'IR Spectrum: {formula}')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.xlim(4000, 400)  # IRスペクトルの慣例に従い、高波数(左) -> 低波数(右)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 主要ピークにラベル付け
    peak_indices = []
    # 簡易的なピーク検出（元のデータを使用）
    for i, (freq, intens) in enumerate(zip(frequencies, intensities)):
        if freq > 400 and freq < 4000 and intens > np.max(intensities) * 0.1:
             peak_indices.append(i)

    # 強度順にソートして上位5つを表示
    peak_indices.sort(key=lambda i: intensities[i], reverse=True)
    for i in peak_indices[:5]:
        freq = frequencies[i]
        intens = intensities[i]
        # ガウシアン上の高さを計算
        h = intens  # 近似的にその高さ
        plt.annotate(f'{freq:.0f}', xy=(freq, h), xytext=(freq, h + np.max(intensities)*0.05),
                     ha='center', fontsize=9, arrowprops=dict(arrowstyle='->', color='black', linewidth=0.5))

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    plt.close()
    print(f"IRスペクトル画像を保存: {save_file}")

def main():
    # グローバル変数の状態をローカルにコピー
    has_infrared = HAS_INFRARED

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

    # 分子式を先に取得
    mol_rdkit = Chem.MolFromSmiles(args.smiles)
    if mol_rdkit is None:
        raise ValueError(f"Invalid SMILES: {args.smiles}")
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol_rdkit)
    
    # ファイル名用のSMILESサニタイズ
    safe_smiles = re.sub(r'[\\/:\*\?"<>\|]', '_', args.smiles)
    
    # ログファイルの設定
    # short_report.txt: Pythonプリント文のみ (要約)
    # log_report.txt: すべての計算過程 (Terminal出力 + Pythonプリント文)
    # 命名規則: {SMILES}_{script}_{method}_{basis}_{type}.txt
    base_name = f"{safe_smiles}_opt-freq_B3LYP_{args.basis}"
    short_log_name = f"{base_name}_short_report.txt"
    full_log_name = f"{base_name}_log_report.txt"

    f_short = open(short_log_name, "w")
    f_full = open(full_log_name, "w")

    # sys.stdoutを置き換え: Terminal + Short + Full
    # これにより print() 文は全て3箇所に出力される
    original_stdout = sys.stdout
    sys.stdout = MultiWriter([original_stdout, f_short, f_full])

    # PySCFの出力先設定: Terminal + Full (Shortには出さない)
    pyscf_writer = MultiWriter([original_stdout, f_full])
    
    print(f"SMILES: {args.smiles}")
    print(f"Method: B3LYP/{args.basis}")
    print(f"要約ログ: {short_log_name}")
    print(f"詳細ログ: {full_log_name}")
    
    try:
        with tqdm(total=5, desc="Overall Progress", file=original_stdout) as pbar:
            pbar.set_description("[1/5] 初期3D構造生成")
            atoms, init_coords = smiles_to_xyz(args.smiles)
            print(f"分子式: {formula}, 原子数: {len(atoms)}")
            pbar.update(1)

            pbar.set_description("[2/5] PySCF分子オブジェクト作成")
            # PySCFの詳細ログは pyscf_writer に出力
            mol = create_mol(atoms, init_coords, args.basis, args.charge, args.spin, output_stream=pyscf_writer)
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

            # Ensure we have a CPU MF object for IR calculation
            if 'mf_cpu' not in locals():
                if hasattr(mf_opt, 'to_cpu'):
                    mf_cpu = mf_opt.to_cpu()
                else:
                    mf_cpu = mf_opt

            # IRスペクトル計算（双極子微分の計算）
            print("   Computing IR intensities...")

            if has_infrared:
                try:
                    # IR計算オブジェクト作成
                    if isinstance(mf_cpu, (dft.rks.RKS, dft.uks.UKS)):
                        ir = infrared.RKS(mf_cpu)
                    else:
                        ir = infrared.RHF(mf_cpu)

                    # 計算済みのHessianをセット（再計算防止）
                    ir.hessian = hess

                    # IR計算実行
                    freq_info = ir.kernel()
                    frequencies = freq_info['freq_wavenumber']

                    # IR強度取得
                    if hasattr(ir, 'ir_intensity'):
                        ir_intensities = ir.ir_intensity
                    else:
                        ir_intensities = np.zeros_like(frequencies)
                        print("⚠️ IR intensities could not be calculated by module.")
                except Exception as e:
                    print(f"⚠️ Error in IR module: {e}")
                    has_infrared = False

            if not has_infrared:
                # Fallback: Numerical differentiation
                # Use freq_info from existing thermo analysis if IR module failed
                if 'freq_info' not in locals():
                     freq_info = thermo.harmonic_analysis(mol_opt, hess)

                frequencies = freq_info['freq_wavenumber']
                modes = freq_info['norm_mode'] # (n_modes, n_atoms, 3)

                ir_intensities = numerical_ir_intensities(mol_opt, mf_cpu, modes)

            # IRスペクトルデータの保存 (CSV)
            ir_csv_name = f"{base_name}_ir.csv"
            print(f"IRスペクトルデータを保存: {ir_csv_name}")
            with open(ir_csv_name, 'w') as f:
                f.write("Frequency(cm-1),Intensity(km/mol)\n")
                for freq, intens in zip(frequencies, ir_intensities):
                    f.write(f"{freq:.4f},{intens:.4f}\n")

            # IRスペクトル画像の保存 (PNG)
            ir_png_name = f"{base_name}_ir.png"
            try:
                plot_ir_spectrum(frequencies, ir_intensities, formula, ir_png_name)
            except Exception as e:
                print(f"⚠️ Failed to plot IR spectrum: {e}")

            # ログ出力用テーブル作成
            print("\n   IR Spectrum Summary:")
            print(f"   {'Freq (cm⁻¹)':>12} {'Intensity (km/mol)':>20}")
            print("   " + "-"*35)
            # 強度が高い順または周波数順？通常は周波数順
            for i in range(len(frequencies)):
                # 虚振動や低強度を除外せず全て表示（または主要なもののみ）
                # ここでは出力が長くなりすぎないように、強度が0より大きい実振動を表示
                if frequencies[i] > 0 and ir_intensities[i] > 1.0:
                    print(f"   {frequencies[i]:12.2f} {ir_intensities[i]:20.2f}")
            print("   " + "-"*35 + "\n")

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
        print(f"比較図を {formula}_comparison.png に保存 (未実装)")

        print("\n" + "="*60)
        print("計算完了！")
        print("="*60)

        end_time = time.time()
        duration = end_time - start_time
        print(f"実行時間: {duration:.2f}秒")

    finally:
        f_short.close()
        f_full.close()
        sys.stdout = original_stdout

if __name__ == "__main__":
    main()
