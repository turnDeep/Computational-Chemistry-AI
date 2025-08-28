#!/usr/bin/env python3
"""
PySCF TD-DFT計算によるUV-Visスペクトル解析
励起状態計算と吸収スペクトル予測

使用例:
python calculate_uv_spectrum.py --smiles "c1ccc2c(c1)ccc1ccccc12"  # アントラセン
python calculate_uv_spectrum.py --smiles "O=C1C=CC(=O)C=C1" --nstates 10  # ベンゾキノン
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pyscf import gto, scf, dft, tdscf
import torch
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

def perform_ground_state(mol, method='B3LYP', use_gpu=False):
    """基底状態計算"""
    
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
    
    # SCF計算
    energy = mf.kernel()
    
    return mf, energy

def perform_tddft(mf, nstates=10):
    """TD-DFT計算"""
    print(f"\nTD-DFT計算中 ({nstates}励起状態)...")
    
    # TD-DFT計算
    td = tdscf.TDDFT(mf)
    td.nstates = nstates
    
    # 励起エネルギーと振動子強度を計算
    e, xy = td.kernel()
    
    # 振動子強度を計算
    td.analyze()
    
    return td, e, xy

def analyze_excitations(td, e, xy):
    """励起状態を解析"""
    # エネルギーをeVとnmに変換
    hartree_to_ev = 27.2114
    ev_to_nm = 1239.84198
    
    excitation_energies_ev = e * hartree_to_ev
    wavelengths_nm = ev_to_nm / excitation_energies_ev
    
    # 振動子強度を取得
    oscillator_strengths = td.oscillator_strength()
    
    results = []
    for i in range(len(e)):
        results.append({
            'state': i + 1,
            'energy_hartree': e[i],
            'energy_ev': excitation_energies_ev[i],
            'wavelength_nm': wavelengths_nm[i],
            'osc_strength': oscillator_strengths[i]
        })
    
    return results

def get_orbital_contributions(td, state_idx):
    """軌道寄与を解析"""
    # 遷移密度行列から主要な軌道遷移を取得
    xy = td.xy[state_idx]
    x, y = xy
    
    # 係数の絶対値でソート
    x_abs = np.abs(x)
    threshold = 0.1  # 10%以上の寄与のみ表示
    
    contributions = []
    nocc = td._scf.mol.nelectron // 2
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x_abs[i, j] > threshold:
                # 占有軌道番号と仮想軌道番号
                occ_idx = i
                vir_idx = nocc + j
                contributions.append({
                    'from': f"HOMO-{nocc-1-i}" if i < nocc else f"occ_{i}",
                    'to': f"LUMO+{j}" if j == 0 else f"LUMO+{j}",
                    'coefficient': x[i, j],
                    'weight': x_abs[i, j]**2
                })
    
    return sorted(contributions, key=lambda x: x['weight'], reverse=True)

def plot_spectrum(results, formula, method, save_file=None):
    """UV-Visスペクトルをプロット"""
    wavelengths = [r['wavelength_nm'] for r in results]
    osc_strengths = [r['osc_strength'] for r in results]
    
    # ガウシアンブロードニング
    x = np.linspace(200, 800, 1000)
    y = np.zeros_like(x)
    sigma = 20  # nm, ブロードニング幅
    
    for wl, osc in zip(wavelengths, osc_strengths):
        gaussian = osc * np.exp(-0.5 * ((x - wl) / sigma) ** 2)
        y += gaussian
    
    # プロット
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # スペクトル
    ax.plot(x, y, 'b-', linewidth=2, label='Simulated spectrum')
    
    # 個々の遷移を棒グラフで表示
    ax.stem(wavelengths, osc_strengths, linefmt='r-', markerfmt='ro', 
            basefmt=' ', label='Transitions')
    
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Oscillator Strength', fontsize=12)
    ax.set_title(f'UV-Vis Spectrum of {formula}\n({method} TD-DFT)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(200, 800)
    ax.set_ylim(0, max(y) * 1.1 if max(y) > 0 else 1)
    
    # 可視光領域を色付け
    ax.axvspan(380, 750, alpha=0.1, color='yellow', label='Visible')
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"スペクトルを {save_file} に保存しました")
    
    plt.show()

def assign_color(wavelength_nm):
    """波長から色を判定"""
    if wavelength_nm < 380:
        return "UV"
    elif wavelength_nm < 450:
        return "紫"
    elif wavelength_nm < 495:
        return "青"
    elif wavelength_nm < 570:
        return "緑"
    elif wavelength_nm < 590:
        return "黄"
    elif wavelength_nm < 620:
        return "橙"
    elif wavelength_nm < 750:
        return "赤"
    else:
        return "近赤外"

def main():
    parser = argparse.ArgumentParser(description='PySCF TD-DFT UV-Visスペクトル計算')
    parser.add_argument('--smiles', type=str, required=True,
                       help='計算対象分子のSMILES')
    parser.add_argument('--method', type=str, default='B3LYP',
                       choices=['HF', 'B3LYP', 'PBE', 'CAM-B3LYP', 'PBE0'],
                       help='計算手法 (default: B3LYP)')
    parser.add_argument('--basis', type=str, default='6-31G*',
                       help='基底関数 (default: 6-31G*)')
    parser.add_argument('--charge', type=int, default=0,
                       help='分子の電荷 (default: 0)')
    parser.add_argument('--spin', type=int, default=0,
                       help='スピン多重度-1 (default: 0)')
    parser.add_argument('--nstates', type=int, default=10,
                       help='計算する励起状態数 (default: 10)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='GPU加速を使用（基底状態計算のみ）')
    parser.add_argument('--plot', action='store_true',
                       help='スペクトルをプロット')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PySCF TD-DFT UV-Visスペクトル計算")
    print("=" * 60)
    print(f"SMILES: {args.smiles}")
    print(f"手法: TD-{args.method}/{args.basis}")
    print(f"励起状態数: {args.nstates}")
    
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
    
    print("\n" + "="*60)
    print("⚠️  警告: 現在の計算は、MMFF力場から生成された初期構造に\n"
          "   基づいています。より正確な結果を得るには、まず同レベルの\n"
          "   理論計算で構造最適化を実行することを強く推奨します。")
    print("="*60)

    # PySCF分子作成
    print("\n[2] PySCF分子オブジェクト作成...")
    mol = create_pyscf_mol(atoms, coords, args.basis, args.charge, args.spin)
    print(f"電子数: {mol.nelectron}")
    print(f"基底関数数: {mol.nao}")
    
    # 基底状態計算
    print(f"\n[3] 基底状態計算 ({args.method}/{args.basis})...")
    mf, energy = perform_ground_state(mol, args.method, args.use_gpu)
    print(f"基底状態エネルギー: {energy:.6f} Hartree")
    
    # HOMO-LUMOギャップ
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    homo_idx = np.where(mo_occ > 0)[0][-1]
    lumo_idx = homo_idx + 1
    gap_ev = (mo_energy[lumo_idx] - mo_energy[homo_idx]) * 27.2114
    print(f"HOMO-LUMO ギャップ: {gap_ev:.2f} eV")
    
    # TD-DFT計算
    print(f"\n[4] TD-{args.method}計算...")
    td, e, xy = perform_tddft(mf, args.nstates)
    
    # 励起状態解析
    print("\n[5] 励起状態解析...")
    results = analyze_excitations(td, e, xy)
    
    # 結果表示
    print("\n" + "=" * 70)
    print(f"{'State':^6} {'Energy (eV)':^12} {'λ (nm)':^10} {'f':^10} {'Color':^10} {'主要遷移':^20}")
    print("-" * 70)
    
    for i, res in enumerate(results):
        color = assign_color(res['wavelength_nm'])
        
        # 最も強い遷移のみ表示
        if res['osc_strength'] > 0.01:  # 振動子強度が0.01以上
            # 軌道寄与を取得（簡略化のため最初の3状態のみ）
            if i < 3:
                contributions = get_orbital_contributions(td, i)
                if contributions:
                    main_contrib = contributions[0]
                    transition = f"{main_contrib['from']}→{main_contrib['to']}"
                else:
                    transition = "複雑"
            else:
                transition = "-"
            
            print(f"S{res['state']:2d}     {res['energy_ev']:8.3f}     "
                  f"{res['wavelength_nm']:7.1f}    {res['osc_strength']:7.4f}   "
                  f"{color:^10} {transition:^20}")
    
    print("=" * 70)
    
    # 最大吸収波長
    max_osc_idx = np.argmax([r['osc_strength'] for r in results])
    max_abs = results[max_osc_idx]
    print(f"\n最大吸収波長: {max_abs['wavelength_nm']:.1f} nm "
          f"({max_abs['energy_ev']:.2f} eV), f = {max_abs['osc_strength']:.4f}")
    print(f"予測される色: {assign_color(max_abs['wavelength_nm'])}")
    
    # 可視光領域の遷移
    visible_transitions = [r for r in results 
                          if 380 <= r['wavelength_nm'] <= 750 
                          and r['osc_strength'] > 0.01]
    if visible_transitions:
        print(f"\n可視光領域の遷移: {len(visible_transitions)}個")
        for trans in visible_transitions[:3]:  # 最初の3つ
            print(f"  λ = {trans['wavelength_nm']:.1f} nm, "
                  f"f = {trans['osc_strength']:.4f}")
    
    # スペクトルプロット
    if args.plot:
        plot_file = f"{formula}_UV-Vis_spectrum.png"
        plot_spectrum(results, formula, args.method, plot_file)
    
    # 結果をファイルに保存
    output_file = f"{formula}_TD-DFT_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"SMILES: {args.smiles}\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Method: TD-{args.method}/{args.basis}\n")
        f.write(f"Ground State Energy: {energy:.6f} Hartree\n")
        f.write(f"HOMO-LUMO Gap: {gap_ev:.2f} eV\n\n")
        f.write("Excitation States:\n")
        f.write(f"{'State':^6} {'Energy (eV)':^12} {'λ (nm)':^10} {'f':^10}\n")
        f.write("-" * 40 + "\n")
        for res in results:
            f.write(f"S{res['state']:2d}     {res['energy_ev']:8.3f}     "
                   f"{res['wavelength_nm']:7.1f}    {res['osc_strength']:7.4f}\n")
        f.write(f"\nMax Absorption: {max_abs['wavelength_nm']:.1f} nm "
               f"(f = {max_abs['osc_strength']:.4f})\n")
    
    print(f"\n結果を {output_file} に保存しました")
    
    print("\n" + "=" * 60)
    print("計算完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()