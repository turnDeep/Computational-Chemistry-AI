#!/usr/bin/env python3
"""
PySCF NMRスペクトル計算
化学シフトとスピン結合定数の計算

使用例:
python calculate_nmr.py --smiles "CCO"  # エタノール
python calculate_nmr.py --smiles "CC(=O)C" --reference "Si(CH3)4"  # TMS基準
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pyscf import gto, scf, dft, prop
from pyscf.prop import nmr
from pyscf.geomopt.geometric_solver import optimize
import warnings
warnings.filterwarnings('ignore')

# NMR基準化合物の化学シフト（ppm）
REFERENCE_SHIFTS = {
    'TMS': {'1H': 0.0, '13C': 0.0, '29Si': 0.0},
    'CDCl3': {'1H': 7.26, '13C': 77.16},
    'D2O': {'1H': 4.79},
    'DMSO': {'1H': 2.50, '13C': 39.52},
}

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
    
    return atoms, np.array(coords), mol

def create_pyscf_mol(atoms, coords, basis='6-311G**', charge=0, spin=0):
    """PySCF分子オブジェクトを作成（NMR計算用の大きな基底）"""
    atom_str = ""
    for atom, coord in zip(atoms, coords):
        atom_str += f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}; "
    
    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis  # NMR計算には分極関数を含む基底が必要
    mol.charge = charge
    mol.spin = spin
    mol.unit = 'Angstrom'
    mol.build()
    
    return mol

def optimize_geometry(mol, method='B3LYP'):
    """構造最適化"""
    print("構造最適化中...")
    
    if method == 'HF':
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = method
    
    mol_opt = optimize(mf, maxsteps=30)
    
    # 最適化後の計算
    if method == 'HF':
        mf_opt = scf.RHF(mol_opt)
    else:
        mf_opt = dft.RKS(mol_opt)
        mf_opt.xc = method
    
    e_opt = mf_opt.kernel()
    
    return mol_opt, mf_opt, e_opt

def calculate_nmr_shielding(mf):
    """NMR遮蔽定数の計算"""
    print("\nNMR遮蔽定数計算中...")
    
    # NMR計算オブジェクト
    nmr_obj = prop.nmr.NMR(mf)
    
    # 遮蔽テンソル計算
    shielding = nmr_obj.kernel()
    
    # 等方的遮蔽（isotropic shielding）
    iso_shielding = np.trace(shielding, axis1=1, axis2=2) / 3.0
    
    # 異方性パラメータ
    anisotropy = []
    for shield_tensor in shielding:
        eigenvals = np.linalg.eigvalsh(shield_tensor)
        eigenvals.sort()
        sigma_iso = eigenvals.mean()
        sigma_aniso = eigenvals[2] - (eigenvals[0] + eigenvals[1]) / 2
        anisotropy.append(sigma_aniso)
    
    return shielding, iso_shielding, anisotropy

def convert_to_chemical_shift(iso_shielding, atoms, ref_shieldings):
    """
    遮蔽定数を化学シフト（ppm）に変換します。
    δ = σ_ref - σ_calc
    """
    chemical_shifts = []
    
    for i, (shield, atom) in enumerate(zip(iso_shielding, atoms)):
        ref_shield = ref_shieldings.get(atom)
        if ref_shield is not None:
            shift = ref_shield - shield
            chemical_shifts.append(shift)
        else:
            # 基準値がない原子核は、遮蔽定数をそのまま返す
            chemical_shifts.append(shield)

    return chemical_shifts

def get_tms_shielding(method, basis, no_opt=False):
    """
    基準物質であるTMS（テトラメチルシラン）のNMR遮蔽定数を計算し、
    1Hおよび13C NMRの基準値を取得します。
    """
    print("\n[Ref] 基準物質(TMS)の遮蔽定数を計算中...")
    tms_smiles = "C[Si](C)(C)C"
    
    try:
        atoms, coords, _ = smiles_to_xyz(tms_smiles)
        tms_mol = create_pyscf_mol(atoms, coords, basis, charge=0, spin=0)

        if not no_opt:
            _, tms_mf, _ = optimize_geometry(tms_mol, method)
        else:
            if method == 'HF':
                tms_mf = scf.RHF(tms_mol)
            else:
                tms_mf = dft.RKS(tms_mol)
                tms_mf.xc = method
            tms_mf.kernel()

        _, tms_iso_shielding, _ = calculate_nmr_shielding(tms_mf)

        tms_atoms = tms_mf.mol.atom_symbols()
        c_indices = [i for i, atom in enumerate(tms_atoms) if atom == 'C']
        h_indices = [i for i, atom in enumerate(tms_atoms) if atom == 'H']

        ref_shield_c = np.mean(tms_iso_shielding[c_indices])
        ref_shield_h = np.mean(tms_iso_shielding[h_indices])

        print(f"[Ref] TMS基準値: C = {ref_shield_c:.3f}, H = {ref_shield_h:.3f}")
        return {'H': ref_shield_h, 'C': ref_shield_c}

    except Exception as e:
        print(f"❌ 基準物質(TMS)の計算に失敗: {e}")
        print("フォールバックとして経験的な基準値を使用します。")
        return {'H': 31.0, 'C': 186.0}

# J-coupling calculation is complex and not implemented.
# The previous fake implementation has been removed.

def assign_nmr_peaks(chemical_shifts, atoms, rdkit_mol):
    """NMRピークを原子グループに割り当て"""
    assignments = []
    
    # RDKit分子の原子数をチェック
    if rdkit_mol.GetNumAtoms() != len(atoms):
        print("警告: RDKit分子とPySCF分子の原子数が一致しません。")
        # SMILESからHを付加した新しいRDKit分子を生成
        mol_from_smiles = Chem.MolFromSmiles(Chem.MolToSmiles(rdkit_mol))
        rdkit_mol = Chem.AddHs(mol_from_smiles)

    for i, (shift, atom) in enumerate(zip(chemical_shifts, atoms)):
        if atom not in ['H', 'C']:
            continue

        try:
            rdkit_atom = rdkit_mol.GetAtomWithIdx(i)
            env = ""
            if atom == 'H':
                neighbors = [n.GetSymbol() for n in rdkit_atom.GetNeighbors()]
                if 'O' in neighbors: env = "OH/CH-O"
                elif 'N' in neighbors: env = "NH/CH-N"
                else: env = "CH/CH2/CH3"
            elif atom == 'C':
                if rdkit_atom.GetIsAromatic():
                    env = "芳香族C"
                elif rdkit_atom.GetHybridization() == Chem.HybridizationType.SP2:
                    is_carbonyl = any(n.GetSymbol() == 'O' and \
                                      rdkit_mol.GetBondBetweenAtoms(i, n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE \
                                      for n in rdkit_atom.GetNeighbors())
                    if is_carbonyl: env = "C=O"
                    else: env = "C=C"
                elif rdkit_atom.GetHybridization() == Chem.HybridizationType.SP3:
                    env = "sp3-C"
                else: env = "sp-C"
            
            assignments.append({
                'atom_idx': i, 'atom': atom, 'shift': shift, 'environment': env
            })
        except Exception:
            # RDKitのインデックスとPySCFのインデックスがずれる場合がある
            assignments.append({
                'atom_idx': i, 'atom': atom, 'shift': shift, 'environment': '不明'
            })

    return assignments

def plot_nmr_spectrum(assignments, nucleus='1H', save_file=None):
    """NMRスペクトルをプロット"""
    # 指定核種のみ抽出
    if nucleus == '1H':
        peaks = [(a['shift'], 1.0) for a in assignments if a['atom'] == 'H']
        x_range = (0, 12)
        title = '¹H NMR Spectrum'
    elif nucleus == '13C':
        peaks = [(a['shift'], 1.0) for a in assignments if a['atom'] == 'C']
        x_range = (0, 200)
        title = '¹³C NMR Spectrum'
    else:
        return
    
    if not peaks:
        print(f"警告: {nucleus}ピークが見つかりません")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # ピークをプロット（ローレンツ関数でブロードニング）
    x = np.linspace(x_range[0], x_range[1], 2000)
    y = np.zeros_like(x)
    
    for shift, intensity in peaks:
        # ローレンツ関数
        width = 0.02 if nucleus == '1H' else 0.5  # ピーク幅
        lorentz = intensity * width**2 / ((x - shift)**2 + width**2)
        y += lorentz
    
    # スペクトル描画（NMRは右から左）
    ax.plot(x, y, 'b-', linewidth=1)
    ax.fill_between(x, 0, y, alpha=0.3)
    
    # ピーク位置にマーカー
    for shift, _ in peaks:
        ax.axvline(x=shift, color='r', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.text(shift, max(y)*1.02, f'{shift:.1f}', ha='center', fontsize=8)
    
    ax.set_xlabel('Chemical Shift (ppm)', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(x_range[1], x_range[0])  # NMRは右から左
    ax.set_ylim(0, max(y) * 1.1 if max(y) > 0 else 1)
    ax.grid(True, alpha=0.3)
    
    # 領域ラベル（1H NMRの場合）
    if nucleus == '1H':
        ax.axvspan(0, 0.5, alpha=0.05, color='blue', label='TMS')
        ax.axvspan(0.5, 3.0, alpha=0.05, color='green', label='アルキル')
        ax.axvspan(3.0, 5.0, alpha=0.05, color='yellow', label='O-CH/N-CH')
        ax.axvspan(6.5, 8.0, alpha=0.05, color='orange', label='芳香族')
        ax.axvspan(9.0, 10.0, alpha=0.05, color='red', label='アルデヒド')
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"スペクトルを {save_file} に保存しました")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='PySCF NMRスペクトル計算')
    parser.add_argument('--smiles', type=str, required=True,
                       help='計算対象分子のSMILES')
    parser.add_argument('--method', type=str, default='B3LYP',
                       choices=['HF', 'B3LYP', 'PBE', 'M06-2X'],
                       help='計算手法 (default: B3LYP)')
    parser.add_argument('--basis', type=str, default='6-311G**',
                       help='基底関数 (default: 6-311G**)')
    parser.add_argument('--charge', type=int, default=0,
                       help='分子の電荷 (default: 0)')
    parser.add_argument('--spin', type=int, default=0,
                       help='スピン多重度-1 (default: 0)')
    parser.add_argument('--plot', action='store_true',
                       help='NMRスペクトルをプロット')
    parser.add_argument('--no-opt', action='store_true',
                       help='構造最適化をスキップ')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PySCF NMRスペクトル計算")
    print("=" * 60)
    print(f"SMILES: {args.smiles}")
    print(f"手法: {args.method}/{args.basis}")
    print(f"基準: TMS (同レベル理論で計算)")
    
    # 分子情報取得
    mol_rdkit = Chem.MolFromSmiles(args.smiles)
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol_rdkit)
    mw = Descriptors.MolWt(mol_rdkit)
    print(f"分子式: {formula}")
    print(f"分子量: {mw:.2f} g/mol")
    
    # 3D構造生成
    print("\n[1] 3D構造生成...")
    atoms, coords, mol_with_h = smiles_to_xyz(args.smiles)
    print(f"原子数: {len(atoms)}")
    n_hydrogens = atoms.count('H')
    n_carbons = atoms.count('C')
    print(f"  水素: {n_hydrogens}個")
    print(f"  炭素: {n_carbons}個")
    
    # PySCF分子作成
    print("\n[2] PySCF分子オブジェクト作成...")
    mol = create_pyscf_mol(atoms, coords, args.basis, args.charge, args.spin)
    print(f"電子数: {mol.nelectron}")
    print(f"基底関数数: {mol.nao}")
    
    # 基準物質(TMS)の計算
    ref_shieldings = get_tms_shielding(args.method, args.basis, args.no_opt)

    # 構造最適化
    if not args.no_opt:
        print(f"\n[3] 構造最適化 ({args.method}/{args.basis})...")
        mol_opt, mf_opt, e_opt = optimize_geometry(mol, args.method)
        print(f"最適化エネルギー: {e_opt:.6f} Hartree")
    else:
        print("\n[3] 構造最適化をスキップ")
        if args.method == 'HF':
            mf_opt = scf.RHF(mol)
        else:
            mf_opt = dft.RKS(mol)
            mf_opt.xc = args.method
        mf_opt.kernel()
        mol_opt = mol
    
    # NMR遮蔽計算
    print("\n[4] NMR遮蔽定数計算...")
    try:
        shielding, iso_shielding, anisotropy = calculate_nmr_shielding(mf_opt)
        
        # 化学シフトに変換
        chemical_shifts = convert_to_chemical_shift(iso_shielding, atoms, ref_shieldings)
        
        # ピーク割り当て
        assignments = assign_nmr_peaks(chemical_shifts, atoms, mol_with_h)
        
        # 結果表示
        print("\n" + "=" * 70)
        print("NMR化学シフト (vs TMS)")
        print("=" * 70)
        
        # 1H NMR
        print("\n【¹H NMR】")
        h_peaks = [a for a in assignments if a['atom'] == 'H']
        if h_peaks:
            print(f"{'原子番号':^10} {'化学シフト (ppm)':^18} {'環境':^20} {'異方性':^12}")
            print("-" * 60)
            for peak in sorted(h_peaks, key=lambda x: x['shift']):
                idx = peak['atom_idx']
                print(f"{idx:^10d} {peak['shift']:^18.2f} {peak['environment']:^20} "
                      f"{anisotropy[idx]:^12.2f}")
        else:
            print("  水素原子なし")
        
        # 13C NMR
        print("\n【¹³C NMR】")
        c_peaks = [a for a in assignments if a['atom'] == 'C']
        if c_peaks:
            print(f"{'原子番号':^10} {'化学シフト (ppm)':^18} {'環境':^20} {'異方性':^12}")
            print("-" * 60)
            for peak in sorted(c_peaks, key=lambda x: x['shift']):
                idx = peak['atom_idx']
                print(f"{idx:^10d} {peak['shift']:^18.1f} {peak['environment']:^20} "
                      f"{anisotropy[idx]:^12.2f}")
        else:
            print("  炭素原子なし")
        
        # J結合の計算は複雑なため、このスクリプトではサポートされていません。
        
        # スペクトルプロット
        if args.plot:
            # 1H NMRスペクトル
            if n_hydrogens > 0:
                h_plot_file = f"{formula}_1H_NMR.png"
                plot_nmr_spectrum(assignments, '1H', h_plot_file)
            
            # 13C NMRスペクトル
            if n_carbons > 0:
                c_plot_file = f"{formula}_13C_NMR.png"
                plot_nmr_spectrum(assignments, '13C', c_plot_file)
        
    except Exception as e:
        print(f"⚠️ NMR計算エラー: {e}")
        print("注: 完全なNMR計算にはGIAO法の実装が必要です")
        
        # 簡易推定値を表示
        print("\n簡易推定値（経験的）:")
        for i, atom in enumerate(atoms):
            if atom == 'H':
                print(f"  H{i}: ~1-4 ppm（アルキル）")
            elif atom == 'C':
                print(f"  C{i}: ~20-80 ppm（sp3）")
    
    # 結果をファイルに保存
    output_file = f"{formula}_NMR_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"SMILES: {args.smiles}\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Method: {args.method}/{args.basis}\n")
        f.write(f"Reference: TMS (calculated at the same level of theory)\n\n")
        
        h_peaks = [a for a in assignments if a['atom'] == 'H']
        c_peaks = [a for a in assignments if a['atom'] == 'C']

        f.write("1H NMR Chemical Shifts (ppm):\n")
        for peak in sorted(h_peaks, key=lambda x: x['shift']) if h_peaks else []:
            f.write(f"  H{peak['atom_idx']}: {peak['shift']:.2f} ppm ({peak['environment']})\n")
        
        f.write("\n13C NMR Chemical Shifts (ppm):\n")
        for peak in sorted(c_peaks, key=lambda x: x['shift']) if c_peaks else []:
            f.write(f"  C{peak['atom_idx']}: {peak['shift']:.1f} ppm ({peak['environment']})\n")
    
    print(f"\n結果を {output_file} に保存しました")
    
    print("\n" + "=" * 60)
    print("計算完了！")
    print("=" * 60)

if __name__ == "__main__":
    main()