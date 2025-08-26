# 🧪 RTX 50シリーズ対応 計算化学・機械学習研究用Docker環境

## 🎮 RTX 5090/5070 Ti完全対応版

オフライン環境での計算化学と機械学習研究に最適化された、**RTX 50シリーズ（Blackwell sm_120）完全対応**のDocker環境です。

## ⚡ RTX 50シリーズサポートの特徴

- **✅ sm_120 (Blackwell) アーキテクチャ完全対応**
- **CUDA 12.8** + **PyTorch Nightly Build (cu128)**
- **RTX 5090** / **RTX 5070 Ti** / **RTX 5080** 動作確認済み
- PyTorchの「sm_120 is not compatible」エラーを解決
- 最新のBlackwell GPUの性能を最大限活用

## 🌟 主な機能

- **Claude Code** + **Ollama** 統合によるローカルLLMエージェント
- **Serena-MCP** によるプログラミング支援
- **RTX 50シリーズ最適化済み** PyTorch環境
- **GPU加速分子計算**: gpu4pyscf-cuda12x対応
- **計算化学ライブラリ完備**: RDKit, ASE, MDAnalysis, PySCF, gpu4pyscf等
- **分子構造最適化**: geomeTRIC統合
- **PubChemデータベースアクセス**: PubChemPy内蔵
- **3D分子可視化**: py3Dmol対応
- **機械学習フレームワーク**: PyTorch (Nightly), TensorFlow, scikit-learn等
- **JupyterLab** 統合開発環境

## 📋 前提条件

- Docker Desktop（WSL2上のUbuntu推奨）
- NVIDIA Docker Runtime（nvidia-container-toolkit）
- **RTX 5090/5070 Ti** または他のRTX 50シリーズGPU
- **NVIDIA Driver 570.xx以上**（CUDA 12.8対応）
- Ollama（ホスト側で稼働中）
- 最低64GB RAM推奨（RTX 5090の場合は128GB推奨）
- 100GB以上の空きディスク容量

## 🚨 重要：RTX 50シリーズ使用時の注意

RTX 50シリーズは新しいBlackwellアーキテクチャ（sm_120）を採用しており、通常のPyTorch安定版では動作しません。この環境は**PyTorch Nightlyビルド**を使用して完全対応しています。

## 🚀 セットアップ手順

### 1. GPUドライバーの確認

```bash
# CUDA 12.8以上が必要
nvidia-smi

# 出力例（RTX 5090の場合）：
# CUDA Version: 12.8
# GPU: NVIDIA GeForce RTX 5090
```

### 2. プロジェクトディレクトリの作成

```bash
mkdir computational-research-rtx50
cd computational-research-rtx50

# 必要なディレクトリ構造を作成
mkdir -p workspace/{notebooks,scripts,data}
mkdir -p config/{claude,serena,claude-bridge}
mkdir -p datasets models logs notebooks
```

### 3. Dockerファイルの配置

このリポジトリの以下のファイルを配置：
- `Dockerfile`（RTX 50シリーズ対応版）
- `docker-compose.yml`（RTX 50シリーズ対応版）
- `requirements.txt`（RTX 50シリーズ用）

### 4. Ollamaモデルの準備（ホスト側）

```bash
# GPT-OSS-20Bモデルを使用
ollama pull gpt-oss-20b

# または他の推奨モデル
ollama pull qwen2.5-coder:7b-instruct
ollama pull deepseek-coder:33b-instruct
```

### 5. Dockerイメージのビルド

```bash
# RTX 50シリーズ対応イメージをビルド（時間がかかります）
docker compose build

# ビルド成功の確認
docker images | grep computational-chemistry-ml
```

### 6. コンテナの起動

```bash
# GPUチェックとメインコンテナの起動
docker compose up -d

# ログでGPU認識を確認
docker compose logs gpu-check
docker compose logs research-env
```

### 7. GPU動作確認

```bash
# コンテナ内でGPU検証スクリプトを実行
docker exec comp-chem-ml-env python3 /usr/local/bin/verify-gpu.py

# 期待される出力：
# ✅ sm_120 (Blackwell) 検出!
# PyTorch Version: 2.x.x+cu128
# ✅ GPU演算テスト成功!
```

### 8. 分子計算環境テスト

```bash
# GPU加速分子計算のテスト
docker exec comp-chem-ml-env python3 /usr/local/bin/test-gpu-chemistry.py

# 期待される出力：
# ✅ gpu4pyscf インストール済み - GPU加速利用可能
# ✅ アスピリン: 分子量=180.16, LogP=1.19
# ✅ PubChem CID: 2244
```

## 💻 使用方法

### JupyterLabへのアクセス

ブラウザで以下にアクセス：
```
http://localhost:8888
Token: research2025
```

### PyTorchでRTX 5090を使用

```python
import torch

# GPU確認
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch version: {torch.__version__}")

# Compute Capability確認（12.0になっているはず）
props = torch.cuda.get_device_properties(0)
print(f"Compute Capability: {props.major}.{props.minor}")

# 大規模テンソル演算（RTX 5090の32GB VRAMを活用）
x = torch.randn(10000, 10000).cuda()
y = torch.randn(10000, 10000).cuda()
z = torch.matmul(x, y)
print(f"演算成功！結果の形状: {z.shape}")
```

### GPU加速分子計算（gpu4pyscf使用）

```python
# GPU加速量子化学計算
import gpu4pyscf
from pyscf import gto

# 分子定義
mol = gto.Mole()
mol.atom = '''
    C  0.0  0.0  0.0
    O  1.2  0.0  0.0
    H -0.5  0.9  0.0
    H -0.5 -0.9  0.0
'''
mol.basis = '6-31G(d)'
mol.build()

# GPU加速Hartree-Fock計算
mf = gpu4pyscf.scf.RHF(mol).to_gpu()
energy = mf.kernel()
print(f"Total Energy (GPU): {energy:.6f} Hartree")

# 分子構造最適化（geometric使用）
from pyscf.geomopt.geometric_solver import optimize
mol_opt = optimize(mf)
print("最適化完了！")
```

### PubChemデータ取得と3D可視化

```python
import pubchempy as pcp
import py3Dmol
from rdkit import Chem

# PubChemから分子情報取得
compounds = pcp.get_compounds('Ibuprofen', 'name')
if compounds:
    smiles = compounds[0].isomeric_smiles
    print(f"SMILES: {smiles}")
    
    # RDKitで3D構造生成
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol)
    
    # py3Dmolで可視化（Jupyter内）
    view = py3Dmol.view(width=400, height=400)
    view.addModel(Chem.MolToMolBlock(mol), 'mol')
    view.setStyle({'stick': {}})
    view.zoomTo()
    view.show()
```

## 🔧 トラブルシューティング

### "sm_120 is not compatible" エラーが出る場合

```bash
# コンテナ内でPyTorchを再インストール
docker exec -it comp-chem-ml-env bash
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### GPUが認識されない場合

```bash
# ホストでNVIDIAランタイムの確認
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi

# Dockerデーモンの設定確認
cat /etc/docker/daemon.json
# "default-runtime": "nvidia" が設定されているか確認
```

## 📊 パフォーマンス最適化（RTX 50向け）

### メモリ最適化

```python
# RTX 5090の32GB VRAMを最大活用
import torch

# メモリアロケータの設定
torch.cuda.set_per_process_memory_fraction(0.9)  # 90%まで使用
torch.cuda.empty_cache()

# Flash Attention 3.0（Blackwell最適化）
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "model_name",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.float16
)
```

### TF32精度の活用

```python
# Blackwellの新しいTensor Coreを活用
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

## 🎮 対応GPU一覧

| GPU | Compute Capability | 状態 |
|-----|-------------------|------|
| RTX 5090 | sm_120 | ✅ 完全対応 |
| RTX 5080 | sm_120 | ✅ 完全対応 |
| RTX 5070 Ti | sm_120 | ✅ 完全対応 |
| RTX 5070 | sm_120 | ✅ 完全対応 |
| RTX 4090 | sm_89 | ✅ 対応（互換性あり） |
| RTX 4080 | sm_89 | ✅ 対応（互換性あり） |

## 📝 技術詳細

- **ベースイメージ**: nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04
- **Python**: 3.11（PyTorch Nightlyとの互換性）
- **PyTorch**: Nightly Build (cu128)
- **CUDA**: 12.8
- **cuDNN**: 9.x（CUDA 12.8に含まれる）
- **CuPy**: 13.6.0（GPU加速計算用）
- **アーキテクチャサポート**: sm_90, sm_120

### 主要計算化学ライブラリ
- **gpu4pyscf-cuda12x**: 1.4.2（GPU加速量子化学計算）
- **PySCF**: 2.5.0（量子化学計算）
- **geometric**: 1.1（分子構造最適化）
- **RDKit**: 2024.03.1（ケモインフォマティクス）
- **PubChemPy**: 1.0.4（PubChemデータベースアクセス）
- **py3Dmol**: 2.5.2（3D分子可視化）

## 🤝 貢献

RTX 50シリーズでの問題や改善案がありましたら、Issueを作成してください。

## 📄 ライセンス

各ライブラリのライセンスに従ってください。

---

**Happy Computing with RTX 50 Series! 🚀🎮🧬💻**