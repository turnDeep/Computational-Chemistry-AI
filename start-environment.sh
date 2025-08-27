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

# Ollamaの接続確認（OpenAI互換エンドポイント）
echo "🔍 Ollama OpenAI互換エンドポイントの確認中..."
if curl -s http://host.docker.internal:11434/v1/models > /dev/null 2>&1; then
    echo "✅ Ollama OpenAI互換エンドポイントに接続成功"
    # 利用可能なモデルの確認
    echo "📋 利用可能なモデル:"
    curl -s http://host.docker.internal:11434/api/tags 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | head -5 || echo "モデル一覧取得をスキップ"
else
    echo "⚠️  警告: Ollamaサーバーに接続できません。"
    echo "    ホスト側で以下を実行してください:"
    echo "    1. ollama serve"
    echo "    2. ollama pull ${OLLAMA_MODEL:-gpt-oss:20b}"
fi

# Claude-bridgeの起動（OpenAI互換モード）
echo "🌉 Claude-bridge（OpenAI互換モード）を起動中..."
echo "   モデル: ${OLLAMA_MODEL:-gpt-oss:20b}"
echo "   API Base: ${OLLAMA_API_BASE:-http://host.docker.internal:11434/v1}"

# Claude-bridgeを起動（バックグラウンド）
OPENAI_API_KEY=dummy claude-bridge openai ${OLLAMA_MODEL:-gpt-oss:20b} \
    --baseURL ${OLLAMA_API_BASE:-http://host.docker.internal:11434/v1} \
    --port ${CLAUDE_BRIDGE_PORT:-8080} \
    > /workspace/logs/claude-bridge.log 2>&1 &
BRIDGE_PID=$!
echo "✅ Claude-bridge起動 (PID: $BRIDGE_PID)"

# 起動確認（3秒待機）
sleep 3
if kill -0 $BRIDGE_PID 2>/dev/null; then
    echo "✅ Claude-bridgeが正常に起動しました"
else
    echo "⚠️  Claude-bridgeの起動に失敗しました。ログを確認してください:"
    echo "    docker exec comp-chem-ml-env cat /workspace/logs/claude-bridge.log"
fi

# Serena-MCPの起動（エラーを無視）
echo "🎯 Serena-MCPサーバーを起動中..."
serena start-mcp-server --context agent --transport sse --port ${MCP_SERVER_PORT:-9121} \
    > /workspace/logs/serena-mcp.log 2>&1 &
SERENA_PID=$!
echo "✅ Serena-MCP起動 (PID: $SERENA_PID) - エラーは無視されます"

# JupyterLabの起動
echo "📊 JupyterLabを起動中..."
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token="${JUPYTER_TOKEN:-research2025}" \
    > /workspace/logs/jupyter.log 2>&1 &
JUPYTER_PID=$!
echo "✅ JupyterLab起動 (PID: $JUPYTER_PID)"

# 起動完了メッセージ
echo ""
echo "=========================================="
echo "🎉 環境の起動が完了しました！"
echo "=========================================="
echo ""
echo "📌 アクセス情報:"
echo "  - JupyterLab: http://localhost:8888"
echo "  - Token: ${JUPYTER_TOKEN:-research2025}"
echo "  - Claude-bridge: http://localhost:8080"
echo ""
echo "🎮 RTX 50シリーズ (sm_120) サポート有効"
echo "🔧 CUDA 12.8 + PyTorch Nightly"
echo "🤖 Ollamaモデル: ${OLLAMA_MODEL:-gpt-oss:20b}"
echo ""
echo "💡 Claude Codeを使用するには:"
echo "  docker exec -it comp-chem-ml-env claude"
echo ""
echo "📁 作業ディレクトリ: /workspace"
echo "📝 ログディレクトリ: /workspace/logs"
echo ""

# プロセスの監視
wait
