# 🧪 RTX 50シリーズ対応 計算化学・機械学習研究用Dev Container環境

## 🎮 RTX 5090/5070 Ti完全対応版 - VS Code Dev Container専用

オフライン環境での計算化学と機械学習研究に最適化された、**RTX 50シリーズ（Blackwell sm_120）完全対応**のVS Code Dev Container環境です。コンテナ内で直接開発が可能な、シンプルで使いやすい構成になっています。

## ⚡ RTX 50シリーズサポートの特徴

- **✅ sm_120 (Blackwell) アーキテクチャ完全対応**
- **CUDA 12.8** + **PyTorch Nightly Build (cu128)**
- **RTX 5090** / **RTX 5070 Ti** / **RTX 5080** 動作確認済み
- PyTorchの「sm_120 is not compatible」エラーを解決
- 最新のBlackwell GPUの性能を最大限活用

## 🌟 主な機能

- **VS Code Dev Container専用設計**: シンプルで使いやすい構成
- **RTX 50シリーズ最適化済み** PyTorch環境
- **GPU加速分子計算**: gpu4pyscf-cuda12x対応
- **計算化学ライブラリ完備**: RDKit, ASE, MDAnalysis, PySCF, gpu4pyscf等
- **分子構造最適化**: geomeTRIC統合
- **PubChemデータベースアクセス**: PubChemPy内蔵
- **3D分子可視化**: py3Dmol対応
- **機械学習フレームワーク**: PyTorch (Nightly), TensorFlow, scikit-learn等
- **JupyterLab**: オプションで利用可能

## 📋 前提条件

- Docker Desktop（WSL2上のUbuntu推奨）
- NVIDIA Docker Runtime（nvidia-container-toolkit）
- **Visual Studio Code** + **Dev Containers拡張機能**（必須）
- **RTX 5090/5070 Ti** または他のRTX 50シリーズGPU
- **NVIDIA Driver 570.xx以上**（CUDA 12.8対応）
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

### 2. リポジトリのクローン

```bash
git clone https://github.com/turnDeep/Computational-Chemistry-AI.git
cd Computational-Chemistry-AI
```

### 3. VS Code Dev Containerで開く

1. **VS Codeに拡張機能をインストール**
   - 拡張機能: `Dev Containers` (ms-vscode-remote.remote-containers)

2. **プロジェクトフォルダを開く**
   ```bash
   code .
   ```

3. **Dev Containerで開く**
   - VS Code左下の緑のアイコンをクリック
   - 「Reopen in Container」を選択
   - 初回はDockerイメージのビルドに**10-15分**かかります

4. **自動GPU検証**
   - コンテナ起動後、自動的にGPU検証スクリプトが実行されます
   - ターミナルで結果を確認してください

### 4. 環境確認

コンテナ内のターミナルで以下を実行：

```bash
# GPU検証
python3 /usr/local/bin/verify-gpu.py

# 期待される出力：
# ✅ sm_120 (Blackwell) 検出!
# PyTorch Version: 2.x.x+cu128
# ✅ GPU演算テスト成功!
```

## 📁 プロジェクト構造

```
Computational-Chemistry-AI/
├── .devcontainer/
│   ├── Dockerfile           # RTX 50シリーズ対応Dockerfile
│   └── devcontainer.json    # Dev Container設定
├── workspace/               # 作業ディレクトリ（コンテナ内 /workspace にマウント）
├── datasets/                # データセット格納用
├── models/                  # モデル保存用
├── logs/                    # ログファイル
├── notebooks/               # Jupyter Notebook
└── README.md
```

## 💻 使用方法

### Dev Container内での開発

コンテナが起動したら、以下が自動的に設定されます：

- ✅ Python環境（/opt/venv）
- ✅ GPU対応PyTorch Nightly
- ✅ 全ての計算化学ライブラリ
- ✅ VS Code Python拡張機能
- ✅ Jupyter Notebook サポート

### Pythonスクリプトの実行

```bash
# ターミナルから直接実行
python your_script.py

# またはVS Codeのデバッガーを使用
# F5キーでデバッグ実行
```

### Jupyter Notebookの使用

**方法1: VS Code内で直接実行（推奨）**
- `.ipynb`ファイルを作成
- VS Code内でそのまま実行可能（JupyterLabサーバー不要）

**方法2: JupyterLabサーバーを起動**
```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# ブラウザで http://localhost:8888 にアクセス
# Token: research2025
```

### コンテナの再起動

```bash
# VS Codeコマンドパレット (Ctrl+Shift+P)
# → "Dev Containers: Rebuild Container"

# またはキャッシュなしで再ビルド
# → "Dev Containers: Rebuild Container Without Cache"
```

## 📝 コード例

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
# コンテナを再ビルド（キャッシュなし）
# VS Code: Ctrl+Shift+P → "Dev Containers: Rebuild Container Without Cache"

# または手動でPyTorchを再インストール
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### GPUが認識されない場合

```bash
# ホストでNVIDIAランタイムの確認
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# Docker Desktopの設定を確認
# Settings → Resources → WSL Integration → Ubuntu を有効化
```

### コンテナビルドが失敗する場合

```bash
# Dockerのリソース制限を増やす
# Docker Desktop → Settings → Resources
# Memory: 16GB以上推奨
# Disk: 100GB以上推奨

# docker-compose関連の古いイメージを削除
docker system prune -a
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

### 環境仕様
- **ベースイメージ**: nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04
- **Python**: 3.11（PyTorch Nightlyとの互換性）
- **PyTorch**: Nightly Build (cu128)
- **CUDA**: 12.8
- **cuDNN**: 9.x（CUDA 12.8に含まれる）
- **CuPy**: 13.4.1（GPU加速計算用）
- **アーキテクチャサポート**: sm_90, sm_120

### 主要計算化学ライブラリ
- **gpu4pyscf**: 最新版（GPU加速量子化学計算、sm_120対応）
- **PySCF**: 2.8.0（量子化学計算）
- **geometric**: 1.1（分子構造最適化）
- **RDKit**: 2024.03.1（ケモインフォマティクス）
- **PubChemPy**: 1.0.4（PubChemデータベースアクセス）
- **py3Dmol**: 2.5.2（3D分子可視化）
- **ASE**: 3.22.1（原子シミュレーション環境）
- **MDAnalysis**: 2.7.0（分子動力学解析）
- **DeepChem**: 最新版（深層学習×化学）

### 機械学習フレームワーク
- **PyTorch**: Nightly (CUDA 12.8)
- **TensorFlow**: 2.16.1
- **Transformers**: 4.40.0
- **scikit-learn**: 1.4.2
- **XGBoost**: 2.0.3
- **LightGBM**: 4.3.0
- **CatBoost**: 1.2.3

## 🆚 従来のdocker-compose版との違い

| 項目 | Dev Container版（現在） | docker-compose版（旧） |
|-----|----------------------|---------------------|
| 設定ファイル数 | 2個（Dockerfile + devcontainer.json） | 3個（Dockerfile + docker-compose.yml + start-environment.sh） |
| 起動方法 | VS Codeから1クリック | docker-compose up コマンド |
| VS Code統合 | ✅ 完全統合 | ⚠️ 手動接続必要 |
| トラブルシューティング | ✅ 簡単 | ⚠️ 複雑 |
| GPU設定 | devcontainer.jsonで管理 | docker-compose.ymlで管理 |

## 🤝 貢献

RTX 50シリーズでの問題や改善案がありましたら、Issueを作成してください。

## 📄 ライセンス

各ライブラリのライセンスに従ってください。

---

**Happy Computing with RTX 50 Series! 🚀🎮🧬💻**
