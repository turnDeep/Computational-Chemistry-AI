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

# Codex CLI はユーザーがフォアグラウンドで実行するため、ここでは何も起動しません。
# 設定は /root/.codex/config.toml で管理されます。

# # Serena-MCPの起動
# echo "🎯 Serenaを起動中（MCPサーバーとダッシュボード）..."
# serena start > /workspace/logs/serena.log 2>&1 &
# SERENA_PID=$!
# echo "✅ Serena起動 (PID: $SERENA_PID)"

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
echo "  - Serena Dashboard: http://localhost:9122"
echo ""
echo "🎮 RTX 50シリーズ (sm_120) サポート有効"
echo "🔧 CUDA 12.8 + PyTorch Nightly"
echo "🤖 Ollamaモデル: ${OLLAMA_MODEL:-gpt-oss:20b}"
echo ""
echo "💡 Codex CLI を使用するには:"
echo "  docker exec -it comp-chem-ml-env codex"
echo ""
echo "📁 作業ディレクトリ: /workspace"
echo "📝 ログディレクトリ: /workspace/logs"
echo ""

# プロセスの監視
wait
