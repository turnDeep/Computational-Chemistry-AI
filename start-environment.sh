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

# 起動完了メッセージ
echo ""
echo "=========================================="
echo "🎉 環境の起動が完了しました！"
echo "=========================================="
echo ""
echo "🎮 RTX 50シリーズ (sm_120) サポート有効"
echo "🔧 CUDA 12.8 + PyTorch Nightly"
echo ""
echo "📁 作業ディレクトリ: /workspace"
echo "📝 ログディレクトリ: /workspace/logs"
echo ""
echo "💡 VS Code Dev Container で開発する場合:"
echo "  VS Code でこのフォルダを開き、'Reopen in Container' を選択"
echo ""
echo "💡 JupyterLab を起動する場合:"
echo "  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root"
echo ""
echo "📌 JupyterLabアクセス情報:"
echo "  - URL: http://localhost:8888"
echo "  - Token: ${JUPYTER_TOKEN:-research2025}"
echo ""

# コンテナを起動し続ける（Dev Container用）
echo "🔄 コンテナを起動状態に保ちます..."
tail -f /dev/null
