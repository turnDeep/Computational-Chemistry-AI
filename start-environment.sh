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

# JupyterLabの起動
echo "📊 JupyterLabを起動中..."
# 仮想環境のPythonを直接指定してJupyterLabを起動する
/opt/venv/bin/python -m jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
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
echo ""
echo "🎮 RTX 50シリーズ (sm_120) サポート有効"
echo "🔧 CUDA 12.8 + PyTorch Nightly"
echo ""
echo "💡 Codex CLI を使用するには:"
echo "  docker exec -it comp-chem-ml-env codex"
echo ""
echo "📁 作業ディレクトリ: /workspace"
echo "📝 ログディレクトリ: /workspace/logs"
echo ""

# プロセスの監視
wait
