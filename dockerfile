# CUDA 13.0.0 Ubuntu 24.04 ベースイメージ
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-13.0 \
    PATH=/usr/local/cuda-13.0/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    CLAUDE_BRIDGE_PORT=8080 \
    MCP_SERVER_PORT=9121 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# 作業ディレクトリ設定
WORKDIR /workspace

# システムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    # 基本ツール
    curl wget git vim nano htop tmux screen \
    build-essential cmake gcc g++ gfortran \
    ca-certificates gnupg lsb-release \
    software-properties-common \
    # Python開発環境
    python3.12 python3.12-dev python3.12-venv python3-pip \
    # Node.js用（Claude Code用）
    nodejs npm \
    # 化学計算用依存関係
    libopenblas-dev liblapack-dev libatlas-base-dev \
    libhdf5-dev libnetcdf-dev \
    libboost-all-dev libgsl-dev \
    libfftw3-dev libsuitesparse-dev \
    # 分子構造可視化用
    libgl1-mesa-glx libglu1-mesa \
    libxrender1 libxext6 libxi6 \
    # ネットワークツール（デバッグ用）
    iputils-ping net-tools dnsutils \
    # SSH（必要な場合）
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Python環境のセットアップ
RUN python3.12 -m pip install --upgrade pip setuptools wheel

# 計算化学・機械学習ライブラリのインストール（段階的に）
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

# 機械学習フレームワーク
RUN pip install --no-cache-dir \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121 \
    torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121 \
    torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    tensorflow==2.16.1 \
    scikit-learn==1.4.2 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    catboost==1.2.3 \
    keras==3.1.1

# 深層学習関連
RUN pip install --no-cache-dir \
    transformers==4.40.0 \
    accelerate==0.29.3 \
    datasets==2.19.0 \
    tokenizers==0.19.1 \
    sentencepiece==0.2.0

# 計算化学専門ライブラリ
RUN pip install --no-cache-dir \
    rdkit==2024.03.1 \
    ase==3.22.1 \
    mdanalysis==2.7.0 \
    mdtraj==1.10.0 \
    pyscf==2.5.0 \
    openbabel-wheel==3.1.1.18 \
    chempy==0.8.3 \
    pymol-open-source==3.0.0

# 分子モデリング・創薬関連
RUN pip install --no-cache-dir \
    biopython==1.83 \
    biotite==0.39.0 \
    prody==2.4.1 \
    oddt==0.8 \
    deepchem==2.7.1 \
    molecularnodes==4.0.0

# 量子化学・材料科学
RUN pip install --no-cache-dir \
    qiskit==1.0.2 \
    qiskit-aer==0.14.0 \
    pymatgen==2024.3.1 \
    phonopy==2.22.0 \
    spglib==2.3.1

# データ処理・可視化追加ライブラリ
RUN pip install --no-cache-dir \
    h5py==3.11.0 \
    netCDF4==1.6.5 \
    xarray==2024.3.0 \
    dask==2024.4.2 \
    bokeh==3.4.1 \
    altair==5.3.0 \
    networkx==3.3 \
    graph-tool==2.68

# 統計・最適化
RUN pip install --no-cache-dir \
    statsmodels==0.14.2 \
    sympy==1.12 \
    cvxpy==1.4.3 \
    optuna==3.6.1 \
    hyperopt==0.2.7 \
    ray[tune]==2.10.0

# AutoML・モデル管理
RUN pip install --no-cache-dir \
    mlflow==2.12.1 \
    wandb==0.16.6 \
    tensorboard==2.16.2 \
    autogluon==1.1.0 \
    pycaret==3.3.1

# 追加の有用なツール
RUN pip install --no-cache-dir \
    joblib==1.4.0 \
    tqdm==4.66.2 \
    rich==13.7.1 \
    pytest==8.1.1 \
    black==24.3.0 \
    flake8==7.0.0 \
    mypy==1.9.0 \
    pre-commit==3.7.0

# Node.js最新化とClaude Code関連のインストール
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest

# Claude Codeのインストール
RUN npm install -g @anthropic-ai/claude-code

# Claude-bridgeのセットアップ
RUN git clone https://github.com/guychenya/LLMBridgeClaudeCode.git /opt/claude-bridge && \
    cd /opt/claude-bridge && \
    pip install uv && \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt

# Serena-MCPのインストール
RUN pip install --no-cache-dir \
    git+https://github.com/oraios/serena.git

# MCP関連の追加パッケージ
RUN pip install --no-cache-dir \
    mcp-server-sqlite \
    duckduckgo-mcp-server

# Ollama-MCP-Bridgeのセットアップ
RUN git clone https://github.com/patruff/ollama-mcp-bridge.git /opt/ollama-mcp-bridge && \
    cd /opt/ollama-mcp-bridge && \
    npm install && \
    npm run build

# 設定ファイルの作成
# Claude-bridge設定
RUN mkdir -p /root/.claude-bridge
COPY <<EOF /root/.claude-bridge/config.json
{
  "ollama": {
    "baseUrl": "http://host.docker.internal:11434",
    "model": "qwen2.5-coder:7b-instruct",
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
RUN mkdir -p /root/.serena
COPY <<EOF /root/.serena/serena_config.yml
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
RUN mkdir -p /root/.config/claude
COPY <<EOF /root/.config/claude/config.json
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

# 起動スクリプトの作成
COPY <<'SCRIPT' /usr/local/bin/start-environment.sh
#!/bin/bash
set -e

echo "🚀 計算化学・機械学習研究環境を起動しています..."

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
cd /opt/claude-bridge
source .venv/bin/activate
python -m llm_bridge_claude_code &
BRIDGE_PID=$!
echo "✅ Claude-bridge起動 (PID: $BRIDGE_PID)"

# Serena-MCPの起動
echo "🎯 Serena-MCPサーバーを起動中..."
serena start-mcp-server --context agent --transport sse --port 9121 &
SERENA_PID=$!
echo "✅ Serena-MCP起動 (PID: $SERENA_PID)"

# JupyterLabの起動
echo "📊 JupyterLabを起動中..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
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
echo "💡 Claude Codeを使用するには:"
echo "  docker exec -it <container-name> claude"
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