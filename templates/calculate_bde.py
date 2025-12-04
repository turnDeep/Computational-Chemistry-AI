#!/usr/bin/env python3
"""
PySCF結合解離エネルギー(BDE)計算
RTX 50シリーズGPU対応版

化合物の全結合のBDE(Bond Dissociation Energy)を計算します。
BDE-db2データベースと同じM06-2X/def2-TZVP手法を使用。

使用例:
python calculate_bde.py --smiles "CCO"  # エタノール
python calculate_bde.py --smiles "CC(=O)O" --use-gpu  # 酢酸（GPU加速）
python calculate_bde.py --smiles "c1ccccc1" --method "B3LYP" --basis "6-31G*"

参考文献:
- BDE-db: https://github.com/nsf-c-cas/BDE-db
- BDE-db2: https://github.com/patonlab/bde-db2
- gpu4pyscf: https://github.com/pyscf/gpu4pyscf
"""

import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from pyscf import gto, scf, dft
import torch
import warnings
import time
from typing import List, Tuple, Dict
warnings.filterwarnings('ignore')

def smiles_to_xyz(smiles: str, optimize: bool = True) -> Tuple[List[str], np.ndarray]:
    """
    SMILESから3D座標を生成

    Args:
        smiles: SMILES記法の分子
        optimize: MMFFで構造最適化するか

    Returns:
        atoms: 原子記号のリスト
        coords: 座標配列
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    # 3D座標を生成
    result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result != 0:
        print("⚠️ 3D座標生成に失敗しました。再試行中...")
        AllChem.EmbedMolecule(mol, randomSeed=0, useRandomCoords=True)

    if optimize:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    conf = mol.GetConformer()
    atoms = []
    coords = []

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(atom.GetSymbol())
        coords.append([pos.x, pos.y, pos.z])

    return atoms, np.array(coords)

def create_pyscf_mol(atoms: List[str], coords: np.ndarray,
                     basis: str = 'def2-TZVP', charge: int = 0,
                     spin: int = 0) -> gto.Mole:
    """
    PySCF分子オブジェクトを作成

    Args:
        atoms: 原子記号のリスト
        coords: 座標配列
        basis: 基底関数
        charge: 電荷
        spin: スピン多重度-1

    Returns:
        mol: PySCF分子オブジェクト
    """
    atom_str = ""
    for atom, coord in zip(atoms, coords):
        atom_str += f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}; "

    mol = gto.Mole()
    mol.atom = atom_str
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.build()

    return mol

def perform_calculation(mol: gto.Mole, method: str = 'M06-2X',
                       use_gpu: bool = False, verbose: int = 0) -> float:
    """
    エネルギー計算を実行

    Args:
        mol: PySCF分子オブジェクト
        method: 計算手法
        use_gpu: GPU加速を使用するか
        verbose: 詳細度

    Returns:
        energy: 全エネルギー（Hartree）
    """

    # GPU利用可能性チェック
    if use_gpu and torch.cuda.is_available():
        try:
            import gpu4pyscf

            if method == 'HF':
                mf = gpu4pyscf.scf.UHF(mol).to_gpu()
            else:
                mf = gpu4pyscf.dft.UKS(mol).to_gpu()
                mf.xc = method
            mf.verbose = verbose
        except ImportError:
            if verbose > 0:
                print("⚠️ gpu4pyscf未インストール、CPUを使用")
            use_gpu = False

    if not use_gpu:
        if method == 'HF':
            mf = scf.UHF(mol)
        else:
            mf = dft.UKS(mol)
            mf.xc = method
        mf.verbose = verbose

    # 収束設定
    mf.conv_tol = 1e-6
    mf.max_cycle = 100

    # エネルギー計算
    try:
        energy = mf.kernel()
        if not mf.converged:
            print("⚠️ SCF計算が収束しませんでした")
            return None
    except Exception as e:
        print(f"⚠️ 計算エラー: {e}")
        return None

    return energy

def get_all_bonds(smiles: str) -> List[Tuple[int, int, str]]:
    """
    分子の全結合を取得

    Args:
        smiles: SMILES記法の分子

    Returns:
        bonds: [(atom1_idx, atom2_idx, bond_type), ...]
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    bonds = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        bond_type = str(bond.GetBondType())

        atom1 = mol.GetAtomWithIdx(idx1).GetSymbol()
        atom2 = mol.GetAtomWithIdx(idx2).GetSymbol()

        bonds.append((idx1, idx2, bond_type, atom1, atom2))

    return bonds, mol

def create_radical_fragments(smiles: str, bond_idx1: int, bond_idx2: int) -> Tuple[str, str]:
    """
    結合を切断してラジカル断片を作成

    Args:
        smiles: SMILES記法の分子
        bond_idx1: 結合の原子1のインデックス
        bond_idx2: 結合の原子2のインデックス

    Returns:
        fragment1_atoms: 断片1の原子リスト
        fragment1_coords: 断片1の座標
        fragment2_atoms: 断片2の原子リスト
        fragment2_coords: 断片2の座標
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # 3D構造を生成
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)

    # 結合を切断
    em = Chem.EditableMol(mol)
    em.RemoveBond(bond_idx1, bond_idx2)
    mol_broken = em.GetMol()

    # 断片を取得
    frags = Chem.GetMolFrags(mol_broken, asMols=True, sanitizeFrags=False)

    if len(frags) != 2:
        return None, None, None, None

    # 各断片の原子と座標を取得
    conf = mol.GetConformer()

    fragments_data = []
    for frag in frags:
        atoms = []
        coords = []
        for atom in frag.GetAtoms():
            orig_idx = atom.GetIntProp('_FromAtomIdx') if atom.HasProp('_FromAtomIdx') else atom.GetIdx()
            pos = conf.GetAtomPosition(orig_idx)
            atoms.append(atom.GetSymbol())
            coords.append([pos.x, pos.y, pos.z])
        fragments_data.append((atoms, np.array(coords)))

    frag1_atoms, frag1_coords = fragments_data[0]
    frag2_atoms, frag2_coords = fragments_data[1]

    return frag1_atoms, frag1_coords, frag2_atoms, frag2_coords

def calculate_bde(smiles: str, bond_idx1: int, bond_idx2: int,
                 method: str = 'M06-2X', basis: str = 'def2-TZVP',
                 use_gpu: bool = False) -> Dict:
    """
    特定の結合のBDEを計算

    BDE = E(radical1) + E(radical2) - E(parent)

    Args:
        smiles: SMILES記法の分子
        bond_idx1: 結合の原子1のインデックス
        bond_idx2: 結合の原子2のインデックス
        method: 計算手法
        basis: 基底関数
        use_gpu: GPU加速を使用するか

    Returns:
        result: BDE計算結果の辞書
    """
    result = {
        'bond': (bond_idx1, bond_idx2),
        'parent_energy': None,
        'fragment1_energy': None,
        'fragment2_energy': None,
        'bde_hartree': None,
        'bde_kcalmol': None,
        'bde_kjmol': None,
        'bde_ev': None,
        'success': False
    }

    try:
        # 親分子のエネルギー計算
        atoms, coords = smiles_to_xyz(smiles)
        mol_parent = create_pyscf_mol(atoms, coords, basis, charge=0, spin=0)

        # 閉殻の親分子にはRKSを使用
        if method == 'HF':
            if use_gpu:
                import gpu4pyscf
                mf_parent = gpu4pyscf.scf.RHF(mol_parent).to_gpu()
            else:
                mf_parent = scf.RHF(mol_parent)
        else:
            if use_gpu:
                import gpu4pyscf
                mf_parent = gpu4pyscf.dft.RKS(mol_parent).to_gpu()
            else:
                mf_parent = dft.RKS(mol_parent)
            mf_parent.xc = method

        mf_parent.verbose = 0
        mf_parent.conv_tol = 1e-6
        mf_parent.max_cycle = 100

        parent_energy = mf_parent.kernel()
        if not mf_parent.converged:
            print(f"  ⚠️ 親分子のSCF計算が収束しませんでした")
            return result

        result['parent_energy'] = parent_energy

        # ラジカル断片を作成
        frag1_atoms, frag1_coords, frag2_atoms, frag2_coords = create_radical_fragments(
            smiles, bond_idx1, bond_idx2
        )

        if frag1_atoms is None:
            print(f"  ⚠️ 断片の作成に失敗しました")
            return result

        # 断片1のエネルギー計算（ラジカル: spin=1）
        mol_frag1 = create_pyscf_mol(frag1_atoms, frag1_coords, basis, charge=0, spin=1)
        frag1_energy = perform_calculation(mol_frag1, method, use_gpu, verbose=0)

        if frag1_energy is None:
            return result

        result['fragment1_energy'] = frag1_energy

        # 断片2のエネルギー計算（ラジカル: spin=1）
        mol_frag2 = create_pyscf_mol(frag2_atoms, frag2_coords, basis, charge=0, spin=1)
        frag2_energy = perform_calculation(mol_frag2, method, use_gpu, verbose=0)

        if frag2_energy is None:
            return result

        result['fragment2_energy'] = frag2_energy

        # BDE計算
        bde_hartree = frag1_energy + frag2_energy - parent_energy
        result['bde_hartree'] = bde_hartree
        result['bde_kcalmol'] = bde_hartree * 627.509  # Hartree to kcal/mol
        result['bde_kjmol'] = bde_hartree * 2625.5     # Hartree to kJ/mol
        result['bde_ev'] = bde_hartree * 27.2114       # Hartree to eV
        result['success'] = True

    except Exception as e:
        print(f"  ⚠️ BDE計算エラー: {e}")

    return result

def main():
    parser = argparse.ArgumentParser(
        description='PySCF結合解離エネルギー(BDE)計算',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python calculate_bde.py --smiles "CCO"
  python calculate_bde.py --smiles "CC(=O)O" --use-gpu
  python calculate_bde.py --smiles "c1ccccc1" --method B3LYP --basis 6-31G*

参考:
  BDE-db2データベースではM06-2X/def2-TZVPが使用されています。
  GPU加速にはgpu4pyscfが必要です。
        """
    )
    parser.add_argument('--smiles', type=str, required=True,
                       help='計算対象分子のSMILES')
    parser.add_argument('--method', type=str, default='M06-2X',
                       choices=['HF', 'B3LYP', 'PBE', 'M06-2X', 'M06', 'wB97X-D'],
                       help='計算手法 (default: M06-2X, BDE-db2と同じ)')
    parser.add_argument('--basis', type=str, default='def2-TZVP',
                       help='基底関数 (default: def2-TZVP, BDE-db2と同じ)')
    parser.add_argument('--use-gpu', action='store_true',
                       help='GPU加速を使用')
    parser.add_argument('--output', type=str, default=None,
                       help='出力ファイル名')

    args = parser.parse_args()

    print("=" * 70)
    print("PySCF 結合解離エネルギー(BDE)計算")
    print("=" * 70)
    print(f"SMILES: {args.smiles}")
    print(f"手法: {args.method}/{args.basis}")

    # GPU情報
    if args.use_gpu:
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️ GPUが利用できません。CPUを使用します。")
            args.use_gpu = False

    # 分子情報取得
    mol_rdkit = Chem.MolFromSmiles(args.smiles)
    if mol_rdkit is None:
        print(f"❌ 無効なSMILES: {args.smiles}")
        return

    mol_rdkit = Chem.AddHs(mol_rdkit)
    formula = rdMolDescriptors.CalcMolFormula(mol_rdkit)
    mw = Descriptors.MolWt(mol_rdkit)
    print(f"分子式: {formula}")
    print(f"分子量: {mw:.2f}")

    # 結合情報を取得
    print("\n[1] 結合情報を取得中...")
    bonds, mol_with_h = get_all_bonds(args.smiles)
    print(f"総結合数: {len(bonds)}")

    # 結合タイプの統計
    bond_types = {}
    for bond in bonds:
        bond_key = f"{bond[3]}-{bond[4]}"
        bond_types[bond_key] = bond_types.get(bond_key, 0) + 1

    print("\n結合タイプの分布:")
    for bond_type, count in sorted(bond_types.items()):
        print(f"  {bond_type}: {count}個")

    # BDE計算
    print(f"\n[2] 各結合のBDE計算中 ({args.method}/{args.basis})...")
    print("-" * 70)

    results = []
    start_time = time.time()

    for i, (idx1, idx2, bond_type, atom1, atom2) in enumerate(bonds, 1):
        print(f"\n結合 {i}/{len(bonds)}: {atom1}({idx1})-{atom2}({idx2}) [{bond_type}]")

        result = calculate_bde(
            args.smiles, idx1, idx2,
            method=args.method,
            basis=args.basis,
            use_gpu=args.use_gpu
        )

        if result['success']:
            print(f"  ✅ BDE = {result['bde_kcalmol']:.2f} kcal/mol "
                  f"({result['bde_kjmol']:.2f} kJ/mol)")
            results.append({
                'bond_id': i,
                'atom1': atom1,
                'atom2': atom2,
                'atom1_idx': idx1,
                'atom2_idx': idx2,
                'bond_type': bond_type,
                **result
            })
        else:
            print(f"  ❌ 計算失敗")

    elapsed_time = time.time() - start_time

    # 結果サマリー
    print("\n" + "=" * 70)
    print("BDE計算結果サマリー")
    print("=" * 70)

    if results:
        print(f"\n成功: {len(results)}/{len(bonds)} 結合\n")

        # 結合タイプごとにグループ化
        bond_type_results = {}
        for r in results:
            key = f"{r['atom1']}-{r['atom2']}"
            if key not in bond_type_results:
                bond_type_results[key] = []
            bond_type_results[key].append(r['bde_kcalmol'])

        # 統計情報を表示
        print(f"{'結合タイプ':<15} {'平均BDE':>12} {'最小BDE':>12} {'最大BDE':>12} {'個数':>6}")
        print("-" * 70)
        for bond_type in sorted(bond_type_results.keys()):
            bdes = bond_type_results[bond_type]
            avg_bde = np.mean(bdes)
            min_bde = np.min(bdes)
            max_bde = np.max(bdes)
            count = len(bdes)
            print(f"{bond_type:<15} {avg_bde:>10.2f} {min_bde:>12.2f} {max_bde:>12.2f} {count:>6}")

        print("\n" + "-" * 70)
        print("\n詳細な結合ごとのBDE:")
        print(f"\n{'ID':<4} {'結合':<15} {'結合タイプ':<10} {'BDE (kcal/mol)':>15} {'BDE (kJ/mol)':>15}")
        print("-" * 70)

        # BDEでソート
        sorted_results = sorted(results, key=lambda x: x['bde_kcalmol'])

        for r in sorted_results:
            bond_label = f"{r['atom1']}({r['atom1_idx']})-{r['atom2']}({r['atom2_idx']})"
            print(f"{r['bond_id']:<4} {bond_label:<15} {r['bond_type']:<10} "
                  f"{r['bde_kcalmol']:>15.2f} {r['bde_kjmol']:>15.2f}")

        # 最弱結合と最強結合
        weakest = sorted_results[0]
        strongest = sorted_results[-1]

        print("\n" + "=" * 70)
        print(f"⚠️  最弱結合: {weakest['atom1']}({weakest['atom1_idx']})-"
              f"{weakest['atom2']}({weakest['atom2_idx']}) "
              f"= {weakest['bde_kcalmol']:.2f} kcal/mol")
        print(f"💪 最強結合: {strongest['atom1']}({strongest['atom1_idx']})-"
              f"{strongest['atom2']}({strongest['atom2_idx']}) "
              f"= {strongest['bde_kcalmol']:.2f} kcal/mol")

    else:
        print("\n❌ すべての計算が失敗しました")

    print(f"\n計算時間: {elapsed_time:.2f} 秒")
    print("=" * 70)

    # 結果をファイルに保存
    output_file = args.output if args.output else f"{formula}_BDE_{args.method}_{args.basis}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"結合解離エネルギー(BDE)計算結果\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"SMILES: {args.smiles}\n")
        f.write(f"分子式: {formula}\n")
        f.write(f"分子量: {mw:.2f}\n")
        f.write(f"手法: {args.method}/{args.basis}\n")
        f.write(f"GPU使用: {'はい' if args.use_gpu else 'いいえ'}\n")
        f.write(f"計算時間: {elapsed_time:.2f} 秒\n")
        f.write(f"\n成功: {len(results)}/{len(bonds)} 結合\n")
        f.write(f"\n{'=' * 70}\n")

        if results:
            # 統計情報
            f.write(f"\n結合タイプ別統計:\n")
            f.write(f"{'-' * 70}\n")
            f.write(f"{'結合タイプ':<15} {'平均BDE':>12} {'最小BDE':>12} {'最大BDE':>12} {'個数':>6}\n")
            f.write(f"{'-' * 70}\n")
            for bond_type in sorted(bond_type_results.keys()):
                bdes = bond_type_results[bond_type]
                avg_bde = np.mean(bdes)
                min_bde = np.min(bdes)
                max_bde = np.max(bdes)
                count = len(bdes)
                f.write(f"{bond_type:<15} {avg_bde:>10.2f} {min_bde:>12.2f} {max_bde:>12.2f} {count:>6}\n")

            # 詳細データ
            f.write(f"\n{'=' * 70}\n")
            f.write(f"結合ごとの詳細データ:\n")
            f.write(f"{'-' * 70}\n")
            f.write(f"{'ID':<4} {'結合':<20} {'結合タイプ':<12} {'BDE(kcal/mol)':>15} "
                   f"{'BDE(kJ/mol)':>15} {'BDE(eV)':>10}\n")
            f.write(f"{'-' * 70}\n")

            for r in sorted(results, key=lambda x: x['bond_id']):
                bond_label = f"{r['atom1']}({r['atom1_idx']})-{r['atom2']}({r['atom2_idx']})"
                f.write(f"{r['bond_id']:<4} {bond_label:<20} {r['bond_type']:<12} "
                       f"{r['bde_kcalmol']:>15.2f} {r['bde_kjmol']:>15.2f} {r['bde_ev']:>10.4f}\n")

            # 最弱・最強結合
            f.write(f"\n{'=' * 70}\n")
            f.write(f"最弱結合: {weakest['atom1']}({weakest['atom1_idx']})-"
                   f"{weakest['atom2']}({weakest['atom2_idx']}) "
                   f"= {weakest['bde_kcalmol']:.2f} kcal/mol\n")
            f.write(f"最強結合: {strongest['atom1']}({strongest['atom1_idx']})-"
                   f"{strongest['atom2']}({strongest['atom2_idx']}) "
                   f"= {strongest['bde_kcalmol']:.2f} kcal/mol\n")

    print(f"\n結果を {output_file} に保存しました")

    # CSV出力も作成
    csv_file = output_file.replace('.txt', '.csv')
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Bond_ID,Atom1,Atom1_Idx,Atom2,Atom2_Idx,Bond_Type,"
               "BDE_Hartree,BDE_kcalmol,BDE_kJmol,BDE_eV\n")
        for r in sorted(results, key=lambda x: x['bond_id']):
            f.write(f"{r['bond_id']},{r['atom1']},{r['atom1_idx']},{r['atom2']},"
                   f"{r['atom2_idx']},{r['bond_type']},"
                   f"{r['bde_hartree']:.6f},{r['bde_kcalmol']:.4f},"
                   f"{r['bde_kjmol']:.4f},{r['bde_ev']:.6f}\n")

    print(f"CSV形式のデータを {csv_file} に保存しました")

if __name__ == "__main__":
    main()
