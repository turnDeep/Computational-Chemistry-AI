#!/usr/bin/env python3
"""
PySCF基本エネルギー計算と分子軌道解析
RTX 50シリーズGPU対応版

使用例:
python calculate_energy.py --smiles "CC(=O)O"  # 酢酸
python calculate_energy.py --smiles "c1ccccc1" --method "B3LYP" --basis "6-31G*"
"""

import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from pyscf import gto, scf, dft
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
    AllChem.MMFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    atoms = []
    coords = []
    
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        atoms.append(atom.GetSymbol())
        coords.append([pos.x, pos.y, pos.z])
    
    return atoms, np.array(coords)

def create_pyscf_mol(atoms, coords, basis='6-31G', charge=0, spin=0):
    """PySCF分子オブジェクトを作成"""
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

def perform_calculation(mol, method='HF', use_gpu=False):
    """エネルギー計算を実行"""
    
    mf = None
    mp2 = None
    
    # GPU利用可能性チェック
    if use_gpu and torch.cuda.is_available():
        try:
            import gpu4pyscf
            print(f"✅ GPU使用: {torch.cuda.get_device_name(0)}")
            
            if method == 'MP2':
                # MP2の場合はまずHF計算を実行
                print("   Step 1: RHF calculation (GPU)...")
                mf = gpu4pyscf.scf.RHF(mol).to_gpu()
                mf.kernel()
                
                print("   Step 2: MP2 calculation (GPU)...")
                from gpu4pyscf import mp
                mp2 = mp.MP2(mf)
                energy = mp2.kernel()
                return mf, energy
                
            elif method == 'HF':
                mf = gpu4pyscf.scf.RHF(mol).to_gpu()
            elif method in ['B3LYP', 'PBE', 'M06-2X']:
                mf = gpu4pyscf.dft.RKS(mol).to_gpu()
                mf.xc = method
            else:
                print(f"⚠️ GPU未対応の手法: {method}, CPUを使用")
                use_gpu = False
        except ImportError:
            print("⚠️ gpu4pyscf未インストール、CPUを使用")
            use_gpu = False
            
    if not use_gpu:
        print("💻 CPU使用")
        if method == 'MP2':
            # MP2の場合はまずHF計算を実行
            print("   Step 1: RHF calculation (CPU)...")
            mf = scf.RHF(mol)
            mf.kernel()
            
            print("   Step 2: MP2 calculation (CPU)...")
            from pyscf import mp
            mp2 = mp.MP2(mf)
            energy = mp2.kernel()
            return mf, energy

        elif method == 'HF':
            mf = scf.RHF(mol)
        elif method in ['B3LYP', 'PBE', 'M06-2X']:
            mf = dft.RKS(mol)
            mf.xc = method
        else:
            raise ValueError(f"未対応の手法: {method}")
    
    # HF/DFTエネルギー計算
    energy = mf.kernel()
    
    return mf, energy

def analyze_orbitals(mf, mol):
    """分子軌道解析"""
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    
    # HOMO/LUMO
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

def calculate_dipole(mf):
    """双極子モーメント計算"""
    from pyscf.scf import hf
    dm = mf.make_rdm1()
    dipole = mf.dip_moment(mf.mol, dm, unit='Debye')
    return dipole

def main():
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
    
    print("=" * 60)
    print("PySCF エネルギー計算")
    print("=" * 60)
    print(f"SMILES: {args.smiles}")
    print(f"手法: {args.method}/{args.basis}")
    
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
    mol = create_pyscf_mol(atoms, coords, args.basis, args.charge, args.spin)
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
    
    # 結果をファイルに保存
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
    
    print(f"結果を {output_file} に保存しました")

if __name__ == "__main__":
    main()
