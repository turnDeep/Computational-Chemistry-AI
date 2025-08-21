# 🧪 計算化学・機械学習研究用Docker環境

オフライン環境での計算化学と機械学習研究に最適化された、完全統合型のDocker環境です。

## 🌟 主な機能

- **Claude Code** + **Ollama** 統合によるローカルLLMエージェント
- **Serena-MCP** によるプログラミング支援
- **CUDA 12.4** 対応（RTX 5090サポート）
- **計算化学ライブラリ完備**: RDKit, ASE, MDAnalysis, PySCF等
- **機械学習フレームワーク**: PyTorch, TensorFlow, scikit-learn等
- **JupyterLab** 統合開発環境

## 📋 前提条件

- Docker Desktop（WSL2上のUbuntu）
- NVIDIA Docker Runtime（nvidia-container-toolkit）
- RTX 5090 + 最新NVIDIAドライバー（CUDA 12.4以上）
- Ollama（ホスト側で稼働中）
- 最低64GB RAM推奨
- 100GB以上の空きディスク容量

## 🚀 セットアップ手順

### 1. プロジェクトディレクトリの作成

```bash
mkdir computational-research
cd computational-research

# 必要なディレクトリ構造を作成
mkdir -p workspace/{notebooks,scripts,data}
mkdir -p config/{claude,serena,claude-bridge}
mkdir -p datasets models logs notebooks
```

### 2. Dockerファイルの配置

このリポジトリの以下のファイルを配置：
- `Dockerfile`（修正版）
- `docker-compose.yml`（修正版）
- `requirements.txt`（修正版）

### 3. Ollamaモデルの準備（ホスト側）

```bash
# 推奨モデルのダウンロード
ollama pull qwen2.5-coder:7b-instruct
ollama pull deepseek-coder:33b-instruct
ollama pull codellama:13b
ollama pull llama3.1:8b
```

### 4. Dockerイメージのビルド

```bash
# イメージをビルド
docker compose build

# またはオフライン用にイメージを保存
docker save computational-chemistry-ml:latest | gzip > comp-chem-ml.tar.gz
```

### 5. コンテナの起動

```bash
# バックグラウンドで起動
docker compose up -d

# ログを確認
docker compose logs -f research-env
```

## 💻 使用方法

### Claude Codeの使用

```bash
# コンテナ内でClaude Codeを起動
docker exec -it comp-chem-ml-env claude

# プロジェクトディレクトリで作業
cd /workspace/my-project
claude --dangerously-skip-permissions
```

### JupyterLabへのアクセス

ブラウザで以下にアクセス：
```
http://localhost:8888
Token: research2025
```

### Serena-MCPの活用

```python
# Pythonスクリプト内から
import requests

# Serena-MCPにコード分析を依頼
response = requests.post('http://localhost:9121/analyze', 
    json={'code': 'your_code_here'})
```

### 計算化学ワークフロー例

```python
# RDKitを使った分子操作
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')
print(f"分子量: {Descriptors.MolWt(mol)}")

# ASEを使った構造最適化
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT

atoms = Atoms('H2O', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
atoms.calc = EMT()
opt = BFGS(atoms)
opt.run(fmax=0.05)

# PyTorchでの分子特性予測（CUDA 12.4対応）
import torch
import torch.nn as nn

# GPUが利用可能か確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用デバイス: {device}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class MolecularNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = MolecularNet(100).to(device)
```

## 📦 含まれるライブラリ

### 計算化学
- **RDKit** - ケモインフォマティクス
- **ASE** - 原子シミュレーション環境
- **MDAnalysis** - 分子動力学解析
- **PySCF** - 量子化学計算
- **OpenBabel** - 分子フォーマット変換
- **PyMOL** - 分子可視化

### 機械学習
- **PyTorch 2.5.1** - 深層学習（CUDA 12.4対応）
- **TensorFlow** - 機械学習フレームワーク
- **scikit-learn** - 古典的機械学習
- **XGBoost/LightGBM/CatBoost** - 勾配ブースティング
- **Transformers** - 事前学習モデル

### データサイエンス
- **NumPy/SciPy** - 数値計算
- **Pandas** - データ処理
- **Matplotlib/Seaborn/Plotly** - 可視化
- **JupyterLab** - インタラクティブ開発

## 🔧 カスタマイズ

### 環境変数の設定

`docker-compose.yml`で以下を調整可能：

```yaml
environment:
  - JUPYTER_TOKEN=your_secure_token
  - OLLAMA_MODEL=preferred_model_name
  - CUDA_VISIBLE_DEVICES=0,1  # GPU選択
```

### 追加パッケージのインストール

```bash
# コンテナ内で
docker exec -it comp-chem-ml-env bash
pip install additional-package
```

## 🛠️ トラブルシューティング

### GPUが認識されない場合

```bash
# コンテナ内で確認
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
```

### Ollamaに接続できない場合

```bash
# ホスト側で
ollama serve

# WSL2の場合、ファイアウォール設定を確認
```

### メモリ不足の場合

`docker-compose.yml`でリソース制限を調整：

```yaml
deploy:
  resources:
    limits:
      memory: 32G  # 必要に応じて調整
```

## 📊 パフォーマンス最適化

### GPUメモリの効率的な使用

```python
# PyTorchでの混合精度学習
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### 並列処理の活用

```python
# Daskで大規模データ処理
import dask.dataframe as dd
df = dd.read_csv('large_dataset.csv')
result = df.groupby('category').mean().compute()
```

## 🔒 セキュリティ注意事項

- オフライン環境での使用を前提としています
- `--dangerously-skip-permissions`は信頼できるコードでのみ使用
- JupyterLabのトークンは必ず変更してください

## 📝 ライセンス

各ライブラリのライセンスに従ってください。主なライセンス：
- RDKit: BSD 3-Clause
- PyTorch: BSD
- TensorFlow: Apache 2.0

## 🤝 サポート

問題が発生した場合は、以下を確認：
1. Dockerログ: `docker compose logs research-env`
2. システムリソース: `docker stats`
3. GPU状態: `nvidia-smi`

## 🔄 バージョン情報

- **CUDA**: 12.4.1
- **cuDNN**: 9.x (CUDA 12.4に含まれる)
- **PyTorch**: 2.5.1 (CUDA 12.4対応)
- **Ubuntu**: 22.04 LTS

---

**Happy Computing! 🚀🧬💻**
