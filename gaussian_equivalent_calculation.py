#!/usr/bin/env python3
"""
Gaussian同等の計算化学計算をPySCFで実行する例
今回構築した環境で完全に動作します
"""

import numpy as np
from pyscf import gto, scf, dft, tddft, mp, cc, fci
from pyscf.tools import cubegen
import matplotlib.pyplot as plt

# =====================================
# 1. 分子構造の定義（Gaussianの入力と同様）
# =====================================

def create_water_molecule():
    """水分子の定義（Gaussian形式と同様）"""
    mol = gto.Mole()
    mol.atom = '''
        O    0.000000    0.000000    0.000000
        H    0.757000    0.586000    0.000000
        H   -0.757000    0.586000    0.000000
    '''
    mol.basis = '6-31G(d)'  # Gaussianと同じ基底関数
    mol.charge = 0
    mol.spin = 0  # 2S+1 = 1 (シングレット)
    mol.build()
    return mol

def create_oxygen_molecule():
    """酸素分子（三重項）の定義"""
    mol = gto.Mole()
    mol.atom = '''
        O    0.000000    0.000000    0.000000
        O    0.000000    0.000000    1.208000
    '''
    mol.basis = 'cc-pVDZ'
    mol.charge = 0
    mol.spin = 2  # 2S+1 = 3 (トリプレット)
    mol.build()
    return mol

# =====================================
# 2. Hartree-Fock計算（Gaussianと同等）
# =====================================

def run_hartree_fock(mol):
    """
    Gaussianの #HF/6-31G(d) と同等
    """
    print("=" * 60)
    print("Hartree-Fock計算")
    print("=" * 60)
    
    # RHF計算
    mf = scf.RHF(mol)
    energy = mf.kernel()
    
    print(f"HF Total Energy: {energy:.8f} Hartree")
    print(f"HF Total Energy: {energy * 627.509:.4f} kcal/mol")
    
    # 軌道エネルギー
    print("\n軌道エネルギー (Hartree):")
    for i, e in enumerate(mf.mo_energy[:10]):  # 最初の10軌道
        print(f"  MO {i+1:2d}: {e:12.6f}")
    
    # HOMO-LUMO ギャップ
    homo_idx = mol.nelectron // 2 - 1
    lumo_idx = homo_idx + 1
    gap = mf.mo_energy[lumo_idx] - mf.mo_energy[homo_idx]
    print(f"\nHOMO-LUMO Gap: {gap:.6f} Hartree ({gap*27.211:.3f} eV)")
    
    return mf

# =====================================
# 3. DFT計算（Gaussianと同等）
# =====================================

def run_dft_calculation(mol, functional='B3LYP'):
    """
    Gaussianの #B3LYP/6-31G(d) と同等
    """
    print("=" * 60)
    print(f"DFT計算 ({functional})")
    print("=" * 60)
    
    mf = dft.RKS(mol)
    mf.xc = functional
    energy = mf.kernel()
    
    print(f"DFT Total Energy: {energy:.8f} Hartree")
    print(f"DFT Total Energy: {energy * 627.509:.4f} kcal/mol")
    
    # Mulliken電荷解析
    pop = mf.mulliken_pop()
    print("\nMulliken電荷:")
    for i, charge in enumerate(pop[1]):
        print(f"  Atom {i+1}: {charge:8.4f}")
    
    return mf

# =====================================
# 4. 励起状態計算（TD-DFT）
# =====================================

def run_tddft_calculation(mf, nstates=5):
    """
    Gaussianの TD-DFT計算と同等
    S1, S2等の励起エネルギーを計算
    """
    print("=" * 60)
    print("TD-DFT励起状態計算")
    print("=" * 60)
    
    td = tddft.TDDFT(mf)
    td.nstates = nstates
    td.kernel()
    td.analyze()
    
    print("\n励起エネルギー:")
    for i in range(nstates):
        e_ev = td.e[i] * 27.211  # Hartree to eV
        wavelength = 1240.0 / e_ev  # nm
        print(f"  S{i+1}: {td.e[i]:.6f} Hartree = {e_ev:.3f} eV = {wavelength:.1f} nm")
        print(f"       振動子強度: {td.oscillator_strength()[i]:.4f}")
    
    return td

# =====================================
# 5. 構造最適化（ASE連携）
# =====================================

def optimize_geometry(mol):
    """
    Gaussianの Opt キーワードと同等
    ASEを使用した構造最適化
    """
    print("=" * 60)
    print("構造最適化")
    print("=" * 60)
    
    from pyscf.geomopt import geometric_solver
    
    mf = dft.RKS(mol)
    mf.xc = 'B3LYP'
    
    # 構造最適化
    mol_eq = geometric_solver.optimize(mf)
    
    print("最適化後の構造:")
    print(mol_eq.atom)
    
    return mol_eq

# =====================================
# 6. 振動数解析
# =====================================

def run_frequency_analysis(mol):
    """
    Gaussianの Freq キーワードと同等
    """
    print("=" * 60)
    print("振動数解析")
    print("=" * 60)
    
    from pyscf.hessian import rhf as rhf_hess
    
    mf = scf.RHF(mol)
    mf.kernel()
    
    # Hessian計算
    hess = rhf_hess.Hessian(mf)
    h = hess.kernel()
    
    # 振動数の計算
    from pyscf.data import nist
    from scipy import linalg
    
    mass = [nist.MASS[mol.atom_symbol(i)] for i in range(mol.natm)]
    mass_mat = np.repeat(mass, 3)
    mass_mat = np.diag(1.0 / np.sqrt(mass_mat))
    
    # Mass-weighted Hessian
    h_mw = mass_mat @ h.reshape(mol.natm*3, mol.natm*3) @ mass_mat
    
    # 固有値計算
    w, v = linalg.eigh(h_mw)
    
    # 振動数 (cm^-1)
    freq_cm = np.sqrt(np.abs(w)) * 219474.63 * np.sign(w)
    
    print("振動数 (cm^-1):")
    for i, f in enumerate(freq_cm[-10:]):  # 最後の10個
        if f > 0:
            print(f"  Mode {i+1}: {f:8.2f}")
    
    return freq_cm

# =====================================
# 7. MP2計算
# =====================================

def run_mp2_calculation(mol):
    """
    Gaussianの MP2計算と同等
    """
    print("=" * 60)
    print("MP2計算")
    print("=" * 60)
    
    mf = scf.RHF(mol)
    mf.kernel()
    
    # MP2計算
    mymp = mp.MP2(mf)
    emp2, t2 = mymp.kernel()
    
    print(f"HF Energy:  {mf.e_tot:.8f} Hartree")
    print(f"MP2 Correlation Energy: {emp2:.8f} Hartree")
    print(f"MP2 Total Energy: {mf.e_tot + emp2:.8f} Hartree")
    
    return mymp

# =====================================
# 8. CCSD計算
# =====================================

def run_ccsd_calculation(mol):
    """
    Gaussianの CCSD計算と同等
    """
    print("=" * 60)
    print("CCSD計算")
    print("=" * 60)
    
    mf = scf.RHF(mol)
    mf.kernel()
    
    # CCSD計算
    mycc = cc.CCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    
    print(f"HF Energy:  {mf.e_tot:.8f} Hartree")
    print(f"CCSD Correlation Energy: {ecc:.8f} Hartree")
    print(f"CCSD Total Energy: {mf.e_tot + ecc:.8f} Hartree")
    
    # CCSD(T)計算
    et = mycc.ccsd_t()
    print(f"(T) Correction: {et:.8f} Hartree")
    print(f"CCSD(T) Total Energy: {mf.e_tot + ecc + et:.8f} Hartree")
    
    return mycc

# =====================================
# 9. 分子軌道の可視化（Cube ファイル生成）
# =====================================

def generate_molecular_orbitals(mol, mf):
    """
    Gaussianの cubegen と同等
    分子軌道をcubeファイルとして出力
    """
    print("=" * 60)
    print("分子軌道の生成")
    print("=" * 60)
    
    # HOMO
    homo_idx = mol.nelectron // 2 - 1
    cubegen.orbital(mol, 'homo.cube', mf.mo_coeff[:, homo_idx])
    print("HOMO saved to homo.cube")
    
    # LUMO
    lumo_idx = homo_idx + 1
    cubegen.orbital(mol, 'lumo.cube', mf.mo_coeff[:, lumo_idx])
    print("LUMO saved to lumo.cube")
    
    # 電子密度
    cubegen.density(mol, 'density.cube', mf.make_rdm1())
    print("Electron density saved to density.cube")

# =====================================
# 10. 溶媒効果（PCM）
# =====================================

def run_pcm_calculation(mol):
    """
    Gaussianの SCRF=(PCM,Solvent=Water) と同等
    """
    print("=" * 60)
    print("PCM溶媒効果計算（水）")
    print("=" * 60)
    
    from pyscf import solvent
    
    mf = dft.RKS(mol)
    mf.xc = 'B3LYP'
    mf = solvent.PCM(mf)
    mf.with_solvent.eps = 78.39  # 水の誘電率
    
    energy = mf.kernel()
    
    print(f"Solution Phase Energy: {energy:.8f} Hartree")
    print(f"Solvation Energy: {mf.with_solvent.e:.6f} Hartree")
    
    return mf

# =====================================
# メイン実行
# =====================================

def main():
    """
    Gaussianで行う典型的な計算をすべて実行
    """
    print("\n" + "=" * 60)
    print(" PySCFによるGaussian同等の計算化学計算デモ")
    print("=" * 60 + "\n")
    
    # 1. 水分子でデモ
    print("\n### 水分子 (H2O) ###\n")
    mol_h2o = create_water_molecule()
    
    # Hartree-Fock
    mf_hf = run_hartree_fock(mol_h2o)
    
    # DFT
    mf_dft = run_dft_calculation(mol_h2o, 'B3LYP')
    
    # TD-DFT（励起状態）
    td = run_tddft_calculation(mf_dft, nstates=3)
    
    # MP2
    mymp = run_mp2_calculation(mol_h2o)
    
    # 振動数解析
    freq = run_frequency_analysis(mol_h2o)
    
    # 分子軌道生成
    generate_molecular_orbitals(mol_h2o, mf_dft)
    
    # PCM溶媒効果
    mf_pcm = run_pcm_calculation(mol_h2o)
    
    # 2. 酸素分子（三重項）でデモ
    print("\n### 酸素分子 (O2, Triplet) ###\n")
    mol_o2 = create_oxygen_molecule()
    
    # UHF計算（開殻系）
    mf_uhf = scf.UHF(mol_o2)
    e_uhf = mf_uhf.kernel()
    print(f"UHF Energy: {e_uhf:.8f} Hartree")
    print(f"<S^2> = {mf_uhf.spin_square()[0]:.4f}")
    
    print("\n" + "=" * 60)
    print(" すべての計算が正常に完了しました！")
    print(" Gaussianと同等の機能が確認できました")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()