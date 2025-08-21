# CUDA 12.1.1 Ubuntu 22.04 ベースイメージ (PyTorchのバージョンと合わせる)
FROM nvidia/cuda:12.1.1-cudnn-devel-ubuntu22.04

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-12.1 \
    PATH=/usr/local/cuda-12.1/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH \
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
    python3.10 python3.10-dev python3.10-venv python3-pip \
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

# Pythonライブラリのインストール
COPY requirements.txt /workspace/requirements.txt
RUN python3.10 -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /workspace/requirements.txt

# Node.js最新化とClaude Code関連のインストール
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest

# Claude Codeのインストール
RUN npm install -g @anthropic-ai/claude-code

# Claude-bridgeのセットアップ
# uvはrequirements.txtでインストール済み
RUN git clone https://github.com/guychenya/LLMBridgeClaudeCode.git /opt/claude-bridge && \
    cd /opt/claude-bridge && \
    uv venv && \
    . .venv/bin/activate && \
    uv pip install -r requirements.txt

# Serena-MCPと関連ツールのインストールはrequirements.txtに移動済み

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