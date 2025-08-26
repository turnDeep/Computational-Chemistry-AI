# ================================================
# RTX 50シリーズ（Blackwell sm_120）対応版
# CUDA 12.8 + PyTorch Nightlyビルドを使用
# ================================================

# CUDA 12.8 Ubuntu 24.04 ベースイメージ（Blackwell対応）
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# 環境変数設定（sm_120対応）
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=gpt-oss-20b \
    CLAUDE_BRIDGE_PORT=8080 \
    MCP_SERVER_PORT=9121 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TORCH_CUDA_ARCH_LIST="9.0;12.0" \
    CUDA_LAUNCH_BLOCKING=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 作業ディレクトリ設定
WORKDIR /workspace

# システムパッケージのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    # 基本ツール
    curl wget git vim nano htop tmux screen \
    build-essential cmake gcc g++ gfortran \
    ca-certificates gnupg lsb-release \
    # Python 3.11（PyTorch Nightlyと互換性が良い）
    python3.11 python3.11-dev python3.11-venv python3-pip \
    # Node.js用（Claude Code用）
    nodejs npm \
    # 化学計算用依存関係
    libopenblas-dev liblapack-dev libatlas-base-dev \
    libhdf5-dev libnetcdf-dev \
    libboost-all-dev libgsl-dev \
    libfftw3-dev libsuitesparse-dev \
    # 分子構造可視化用
    libgl1 libglu1-mesa \
    libxrender1 libxext6 libxi6 \
    # ネットワークツール（デバッグ用）
    iputils-ping net-tools dnsutils \
    # SSH（必要な場合）
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11をデフォルトに設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# pipのアップグレード
RUN python3.11 -m pip install --upgrade pip

# ===================================================
# RTX 50シリーズ対応: PyTorch Nightly (cu128) インストール
# ===================================================
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# CuPy for CUDA 12.x（GPU4PySCF用）
RUN pip install --no-cache-dir \
    cupy-cuda12x==13.6.0 \
    cutensor-cu12

# 基本的な科学計算ライブラリ
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.0 \
    pandas==2.2.2 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    plotly==5.20.0 \
    jupyter==1.0.0 \
    jupyterlab==4.1.5 \
    ipython==8.23.0

# 機械学習フレームワーク（PyTorch以外）
RUN pip install --no-cache-dir \
    tensorflow==2.16.1 \
    scikit-learn==1.4.2 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    catboost==1.2.3 \
    keras==3.1.1

# Deep Learning - Transformers & Ecosystem
RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    accelerate==0.29.3 \
    datasets==2.19.0 \
    tokenizers==0.19.1 \
    sentencepiece==0.2.0

# 計算化学ライブラリ（GPU対応版含む）
RUN pip install --no-cache-dir \
    rdkit==2024.03.1 \
    ase==3.22.1 \
    mdanalysis==2.7.0 \
    mdtraj==1.10.0 \
    pyscf==2.5.0 \
    gpu4pyscf-cuda12x==1.4.2 \
    geometric==1.1 \
    pubchempy==1.0.4 \
    py3Dmol==2.5.2 \
    openbabel-wheel==3.1.1.18 \
    chempy==0.8.3

# 分子モデリング
RUN pip install --no-cache-dir \
    biopython==1.83 \
    biotite==0.39.0 \
    prody==2.4.1 \
    oddt==0.8 \
    deepchem==2.7.1

# データ処理とツール
RUN pip install --no-cache-dir \
    h5py==3.11.0 \
    netCDF4==1.6.5 \
    xarray==2024.3.0 \
    dask==2024.4.2 \
    bokeh==3.4.1 \
    altair==5.3.0 \
    networkx==3.3

# 開発ツール
RUN pip install --no-cache-dir \
    joblib==1.4.0 \
    tqdm==4.66.2 \
    rich==13.7.1 \
    pytest==8.1.1 \
    black==24.3.0 \
    flake8==7.0.0 \
    mypy==1.9.0 \
    pre-commit==3.7.0 \
    uv

# Node.js最新化とClaude Code関連のインストール
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest

# Claude Codeのインストール
RUN npm install -g @anthropic-ai/claude-code

# Claude-bridgeのセットアップ
RUN git clone https://github.com/guychenya/LLMBridgeClaudeCode.git /opt/claude-bridge && \
    cd /opt/claude-bridge && \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt

# Serena-MCPのインストール
RUN pip install --no-cache-dir git+https://github.com/oraios/serena.git

# Ollama-MCP-Bridgeのセットアップ
RUN git clone https://github.com/patruff/ollama-mcp-bridge.git /opt/ollama-mcp-bridge && \
    cd /opt/ollama-mcp-bridge && \
    npm install && \
    npm run build || true

# 設定ファイルの作成
# Claude-bridge設定
RUN mkdir -p /root/.claude-bridge && \
    cat <<EOF > /root/.claude-bridge/config.json
{
  "ollama": {
    "baseUrl": "http://host.docker.internal:11434",
    "model": "__OLLAMA_MODEL_PLACEHOLDER__",
    "timeout": 300000
  },
  "server": {
    "port": 8080,
    "host": "0.0.0.0"
  },
  "logging": {
    "level": "info",
    "file": "/workspace/logs/claude-bridge.log"
  }
}
EOF

# Serena-MCP設定
RUN mkdir -p /root/.serena && \
    cat <<EOF > /root/.serena/serena_config.yml
contexts:
  agent:
    system_prompt: |
      You are Serena, a powerful coding agent specialized in computational chemistry and machine learning.
      You have access to comprehensive Python libraries for scientific computing.
    tools:
      - read_file
      - write_file
      - search_symbols
      - edit_code
      - execute_command
      - create_project
      
modes:
  research:
    description: "Research mode for computational chemistry"
    context: agent
    
settings:
  workspace_dir: /workspace
  max_file_size: 10485760
  enable_web_dashboard: true
  dashboard_port: 9122
EOF

# MCP統合設定
RUN mkdir -p /root/.config/claude && \
    cat <<EOF > /root/.config/claude/config.json
{
  "mcpServers": {
    "serena": {
      "command": "serena",
      "args": ["start-mcp-server", "--context", "agent", "--transport", "stdio"]
    },
    "ollama-bridge": {
      "command": "node",
      "args": ["/opt/ollama-mcp-bridge/dist/index.js"],
      "env": {
        "OLLAMA_HOST": "http://host.docker.internal:11434"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
    }
  }
}
EOF

# GPU検証スクリプトの作成
RUN cat <<'SCRIPT' > /usr/local/bin/verify-gpu.py
#!/usr/bin/env python3
import torch
import sys

print("=" * 60)
print("RTX 50シリーズ GPU検証")
print("=" * 60)

# CUDA利用可能性チェック
cuda_available = torch.cuda.is_available()
print(f"CUDA利用可能: {cuda_available}")

if cuda_available:
    # デバイス情報
    device_count = torch.cuda.device_count()
    print(f"GPUデバイス数: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  メモリ: {props.total_memory / 1e9:.1f} GB")
        print(f"  SM数: {props.multi_processor_count}")
        
        # sm_120のチェック
        if props.major == 12 and props.minor == 0:
            print(f"  ✅ sm_120 (Blackwell) 検出!")
    
    # PyTorchバージョン情報
    print(f"\nPyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    
    # 簡単なGPU演算テスト
    try:
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.mm(test_tensor, test_tensor)
        print("\n✅ GPU演算テスト成功!")
    except Exception as e:
        print(f"\n❌ GPU演算テスト失敗: {e}")
        sys.exit(1)
else:
    print("❌ CUDAが利用できません")
    sys.exit(1)

print("=" * 60)
SCRIPT

# GPU分子計算サンプルスクリプトの作成
RUN cat <<'SCRIPT' > /usr/local/bin/test-gpu-chemistry.py
#!/usr/bin/env python3
"""
RTX 50シリーズでのGPU加速分子計算デモ
"""
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import pyscf
from pyscf import gto, scf
import pubchempy as pcp

print("=" * 60)
print("GPU加速分子計算環境テスト")
print("=" * 60)

# 1. GPU確認
print("\n[1] GPU状態確認")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    props = torch.cuda.get_device_properties(0)
    if props.major == 12 and props.minor == 0:
        print(f"✅ RTX 50シリーズ (sm_120) 検出！")
else:
    print("❌ GPU利用不可")

# 2. RDKit分子処理
print("\n[2] RDKit分子処理")
mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')  # アスピリン
if mol:
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    print(f"✅ アスピリン: 分子量={mw:.2f}, LogP={logp:.2f}")

# 3. PySCF量子化学計算
print("\n[3] PySCF量子化学計算")
mol_h2o = gto.Mole()
mol_h2o.atom = '''
    O  0.0  0.0  0.0
    H  0.757  0.586  0.0
    H -0.757  0.586  0.0
'''
mol_h2o.basis = '6-31G'
mol_h2o.build()

# 4. GPU加速の確認（gpu4pyscfがインストールされている場合）
try:
    import gpu4pyscf
    print("✅ gpu4pyscf インストール済み - GPU加速利用可能")
    # GPU加速SCF計算
    mf = gpu4pyscf.scf.RHF(mol_h2o).to_gpu()
    energy = mf.kernel()
    print(f"   HF Energy (GPU): {energy:.6f} Hartree")
except ImportError:
    print("⚠️  gpu4pyscf CPU版を使用")
    mf = scf.RHF(mol_h2o)
    energy = mf.kernel()
    print(f"   HF Energy (CPU): {energy:.6f} Hartree")

# 5. PubChemPyテスト
print("\n[4] PubChemPyデータ取得")
try:
    compounds = pcp.get_compounds('Aspirin', 'name')
    if compounds:
        c = compounds[0]
        print(f"✅ PubChem CID: {c.cid}")
        print(f"   分子式: {c.molecular_formula}")
except Exception as e:
    print(f"⚠️  PubChemアクセスエラー: {e}")

print("\n" + "=" * 60)
print("すべてのテスト完了！")
print("=" * 60)
SCRIPT

RUN chmod +x /usr/local/bin/test-gpu-chemistry.py

# 起動スクリプトの作成
RUN cat <<'SCRIPT' > /usr/local/bin/start-environment.sh
#!/bin/bash
set -e

echo "🚀 RTX 50シリーズ対応 計算化学・機械学習研究環境を起動しています..."

# GPU検証
echo "🎮 GPU検証中..."
python3 /usr/local/bin/verify-gpu.py || {
    echo "⚠️ GPU検証に失敗しましたが、続行します..."
}

# 分子計算環境テスト（オプション）
echo "🧪 分子計算環境の確認中..."
python3 -c "import pyscf, rdkit, pubchempy, py3Dmol; print('✅ 主要ライブラリ利用可能')" || {
    echo "⚠️ 一部のライブラリが利用できません"
}

# ログディレクトリ作成
mkdir -p /workspace/logs

# Ollamaの接続確認
echo "🔍 Ollamaサーバーの接続を確認中..."
if curl -s http://host.docker.internal:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollamaサーバーに接続成功"
else
    echo "⚠️  警告: Ollamaサーバーに接続できません。ホスト側でOllamaが起動していることを確認してください。"
fi

# Claude-bridgeの起動
echo "🌉 Claude-bridgeを起動中..."
# モデル設定を環境変数から反映
sed -i "s|__OLLAMA_MODEL_PLACEHOLDER__|${OLLAMA_MODEL:-gpt-oss-20b}|g" /root/.claude-bridge/config.json
cd /opt/claude-bridge
source .venv/bin/activate
python -m llm_bridge_claude_code &
BRIDGE_PID=$!
echo "✅ Claude-bridge起動 (PID: $BRIDGE_PID)"

# Serena-MCPの起動（エラーを無視）
echo "🎯 Serena-MCPサーバーを起動中..."
serena start-mcp-server --context agent --transport sse --port 9121 2>/dev/null &
SERENA_PID=$!
echo "✅ Serena-MCP起動 (PID: $SERENA_PID)"

# JupyterLabの起動
echo "📊 JupyterLabを起動中..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token="${JUPYTER_TOKEN:-research2025}" &
JUPYTER_PID=$!
echo "✅ JupyterLab起動 (PID: $JUPYTER_PID)"

echo ""
echo "=========================================="
echo "🎉 環境の起動が完了しました！"
echo "=========================================="
echo ""
echo "📌 アクセス情報:"
echo "  - JupyterLab: http://localhost:8888"
echo "  - Claude-bridge: http://localhost:8080"
echo "  - Serena-MCP: http://localhost:9121"
echo "  - Serena Dashboard: http://localhost:9122"
echo ""
echo "🎮 RTX 50シリーズ (sm_120) サポート有効"
echo "🔧 CUDA 12.8 + PyTorch Nightly"
echo ""
echo "💡 Claude Codeを使用するには:"
echo "  docker exec -it comp-chem-ml-env claude"
echo ""
echo "📁 作業ディレクトリ: /workspace"
echo ""

# プロセスの監視
wait
SCRIPT

RUN chmod +x /usr/local/bin/start-environment.sh

# Pythonパス設定
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# ポート公開
EXPOSE 8080 8888 9121 9122

# ボリュームマウントポイント
VOLUME ["/workspace", "/root/.claude", "/root/.serena"]

# エントリーポイント
ENTRYPOINT ["/usr/local/bin/start-environment.sh"]
