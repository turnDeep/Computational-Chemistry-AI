# PySCF計算化学テンプレート集
## RTX 50シリーズGPU対応版

このテンプレート集は、RTX 50シリーズ（RTX 5090/5070 Ti）対応のDocker環境で、SMILESから分子を生成して様々な量子化学計算を実行できるPythonスクリプト集です。

## 📋 テンプレート一覧

### 1. **基本エネルギー計算と分子軌道解析** (`calculate_energy.py`)
分子の全エネルギー、HOMO-LUMO、双極子モーメントを計算

### 2. **構造最適化と振動数解析** (`optimize_geometry.py`)
分子構造の最適化と熱力学的特性の計算

### 3. **TD-DFTによるUV-Visスペクトル計算** (`calculate_uv_spectrum.py`)
励起状態計算と吸収スペクトルの予測

### 4. **IRスペクトル計算と振動モード解析** (`calculate_ir_spectrum.py`)
赤外分光スペクトルと振動モードの解析

### 5. **溶媒効果計算（PCMモデル）** (`calculate_solvent_effect.py`)
様々な溶媒中での安定性と溶媒和エネルギーの計算

### 6. **結合解離エネルギー(BDE)計算** (`calculate_bde.py`)
化合物の全結合のBDEを計算（BDE-db2準拠、M06-2X/def2-TZVP）

## 🚀 使用方法

### 基本的な使い方

すべてのスクリプトは `--smiles` オプションでSMILES記法の分子を指定します：

```bash
# Docker環境内で実行
docker exec -it comp-chem-ml-env bash

# 作業ディレクトリに移動
cd /workspace

# スクリプトを実行
python calculate_energy.py --smiles "CC(=O)O"  # 酢酸の例
```

### 共通オプション

すべてのスクリプトで使用可能：
- `--smiles`: 計算対象分子のSMILES記法（必須）
- `--method`: 計算手法 (HF, B3LYP, PBE, M06-2X)
- `--basis`: 基底関数 (6-31G, 6-31G*, cc-pVDZ等)
- `--charge`: 分子の電荷 (デフォルト: 0)
- `--spin`: スピン多重度-1 (デフォルト: 0)
- `--use-gpu`: GPU加速を使用（RTX 50シリーズ推奨）

## 📚 使用例

### 1. エネルギー計算

```bash
# アスピリンのエネルギー計算
python calculate_energy.py --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --method B3LYP --basis 6-31G* --use-gpu

# 出力例:
# 全エネルギー: -648.123456 Hartree
# HOMO: -0.2345 Hartree
# LUMO: -0.0123 Hartree
# HOMO-LUMO ギャップ: 6.04 eV
# 双極子モーメント: 4.567 Debye
```

### 2. 構造最適化

```bash
# エタノールの構造最適化と振動数解析
python optimize_geometry.py --smiles "CCO" --freq --use-gpu

# 出力例:
# 最適化エネルギー: -154.987654 Hartree
# 虚振動数: 0個 ✅ 安定構造
# ゼロ点エネルギー: 48.2 kcal/mol
# ギブズ自由エネルギー: -154.945 Hartree
```

### 3. UV-Visスペクトル

```bash
# アントラセンのUV-Vis計算（10励起状態）
python calculate_uv_spectrum.py --smiles "c1ccc2cc3ccccc3cc2c1" --nstates 10 --plot

# 出力例:
# 最大吸収波長: 375.2 nm (f = 0.234)
# 予測される色: UV領域
# S1: 3.31 eV, 374.8 nm, f=0.234, HOMO→LUMO
```

### 4. IRスペクトル

```bash
# 酢酸のIRスペクトル計算
python calculate_ir_spectrum.py --smiles "CC(=O)O" --plot

# 出力例:
# C=O伸縮: 1742.3 cm⁻¹ (強度: 285.4)
# O-H伸縮: 3587.2 cm⁻¹ (強度: 147.8)
# C-H伸縮: 2941.5 cm⁻¹ (強度: 42.1)
```

### 5. 溶媒効果

```bash
# ベンゾキノンの溶媒効果比較
python calculate_solvent_effect.py --smiles "O=C1C=CC(=O)C=C1" --compare-solvents

# 出力例:
# 溶媒        ε    ΔGsolv (kcal/mol)  双極子 (D)
# 水       78.4      -12.34           4.56
# エタノール  24.6       -8.91           3.98
# ベンゼン    2.3       -2.13           3.12
```

### 6. BDE計算

```bash
# エタノールの全結合のBDE計算（BDE-db2準拠）
python calculate_bde.py --smiles "CCO" --use-gpu

# 出力例:
# 結合タイプ  平均BDE   最小BDE   最大BDE   個数
# C-C         85.32     85.32     85.32      1
# C-H         98.45     96.12    102.34      5
# C-O         86.71     86.71     86.71      1
# O-H        104.23    104.23    104.23      1
#
# 最弱結合: C(1)-H(4) = 96.12 kcal/mol
# 最強結合: O(2)-H(8) = 104.23 kcal/mol

# 酢酸のBDE計算（カルボン酸のO-H結合の解離）
python calculate_bde.py --smiles "CC(=O)O" --method M06-2X --basis def2-TZVP

# ベンゼンのBDE計算
python calculate_bde.py --smiles "c1ccccc1" --use-gpu
```

## 💻 高度な使用例

### バッチ処理スクリプト

複数の分子を一度に計算する例：

```python
#!/usr/bin/env python3
# batch_calculation.py

import subprocess
import json

# 計算したい分子のリスト
molecules = {
    'methanol': 'CO',
    'ethanol': 'CCO',
    'acetone': 'CC(=O)C',
    'benzene': 'c1ccccc1',
    'toluene': 'Cc1ccccc1',
    'phenol': 'Oc1ccccc1'
}

results = {}

for name, smiles in molecules.items():
    print(f"Calculating {name}...")
    
    # エネルギー計算を実行
    cmd = f"python calculate_energy.py --smiles '{smiles}' --method B3LYP --basis 6-31G*"
    
    try:
        output = subprocess.check_output(cmd, shell=True, text=True)
        # 結果を解析（簡易版）
        for line in output.split('\n'):
            if '全エネルギー:' in line:
                energy = float(line.split(':')[1].split()[0])
                results[name] = energy
                break
    except Exception as e:
        print(f"Error calculating {name}: {e}")
        results[name] = None

# 結果を保存
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("バッチ計算完了！")
```

### パイプライン処理

完全な分子解析パイプライン：

```bash
#!/bin/bash
# full_analysis.sh

SMILES=$1
NAME=${2:-molecule}

echo "========================================="
echo "分子の完全解析: $NAME"
echo "SMILES: $SMILES"
echo "========================================="

# 1. 構造最適化
echo "Step 1: 構造最適化..."
python optimize_geometry.py --smiles "$SMILES" --freq

# 2. エネルギー計算（最適化構造で）
echo "Step 2: エネルギー解析..."
python calculate_energy.py --smiles "$SMILES" --use-gpu

# 3. UV-Visスペクトル
echo "Step 3: UV-Visスペクトル..."
python calculate_uv_spectrum.py --smiles "$SMILES" --nstates 20 --plot

# 4. IRスペクトル
echo "Step 4: IRスペクトル..."
python calculate_ir_spectrum.py --smiles "$SMILES" --plot

# 5. 溶媒効果
echo "Step 5: 溶媒効果..."
python calculate_solvent_effect.py --smiles "$SMILES" --compare-solvents

echo "========================================="
echo "解析完了！結果ファイルを確認してください。"
echo "========================================="
```

使用例：
```bash
chmod +x full_analysis.sh
./full_analysis.sh "CC(=O)O" "acetic_acid"
```

## 🔧 カスタマイズ

### 新しい計算手法の追加

テンプレートを拡張して新しい手法を追加できます：

```python
# CCSD(T)計算の追加例
from pyscf import cc

def perform_ccsd_t(mol):
    """CCSD(T)計算"""
    mf = scf.RHF(mol)
    mf.kernel()
    
    # CCSD計算
    mycc = cc.CCSD(mf)
    mycc.kernel()
    
    # (T)補正
    et = mycc.ccsd_t()
    
    return mycc.e_tot + et
```

### GPU最適化のヒント

RTX 5090の32GB VRAMを最大活用：

```python
# 大規模分子用の設定
import torch

# メモリ管理
torch.cuda.set_per_process_memory_fraction(0.9)
torch.cuda.empty_cache()

# TF32精度の活用（Blackwellアーキテクチャ）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

## 📊 出力ファイル

各スクリプトは以下のファイルを生成：

- `{分子式}_{手法}_{基底}.txt`: 計算結果のサマリー
- `{分子式}_optimized.xyz`: 最適化構造（XYZ形式）
- `{分子式}_UV-Vis_spectrum.png`: UV-Visスペクトル図
- `{分子式}_IR_spectrum.png`: IRスペクトル図
- `{分子式}_solvent_effect.txt`: 溶媒効果の詳細

## ⚠️ 注意事項

1. **計算時間**: 大きな分子や高精度計算は時間がかかります
2. **メモリ使用**: 大規模基底関数はメモリを大量に消費します
3. **収束問題**: 一部の分子では収束しない場合があります
4. **GPU互換性**: gpu4pyscfは一部の計算手法のみGPU対応

## 🆘 トラブルシューティング

### エラー: "Invalid SMILES"
```bash
# SMILES記法を確認し、クォートで囲む
python calculate_energy.py --smiles 'C(C)(C)C'  # 正しい
python calculate_energy.py --smiles C(C)(C)C    # 間違い
```

### エラー: "SCF not converged"
```python
# 収束条件を緩和
mf.conv_tol = 1e-6  # デフォルト: 1e-9
mf.max_cycle = 100  # デフォルト: 50
```

### GPU関連のエラー
```bash
# GPUが認識されているか確認
python -c "import torch; print(torch.cuda.is_available())"

# gpu4pyscfのインストール確認
python -c "import gpu4pyscf; print('OK')"
```

## 📚 参考文献

- PySCF公式ドキュメント: https://pyscf.org/
- GPU4PySCF: https://github.com/pyscf/gpu4pyscf
- BDE-db: https://github.com/nsf-c-cas/BDE-db (290,664 BDE値)
- BDE-db2: https://github.com/patonlab/bde-db2 (531,244 BDE値)
- 量子化学の基礎理論については専門書を参照

## 🎯 今後の拡張予定

- [ ] CASSCF/CASCI計算テンプレート
- [ ] NMRスペクトル計算
- [ ] 反応経路探索
- [ ] 分子動力学シミュレーション
- [ ] 機械学習ポテンシャル

---

**Happy Computing with RTX 50 Series! 🚀🧬💻**