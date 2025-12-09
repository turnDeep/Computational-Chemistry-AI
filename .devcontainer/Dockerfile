# ================================================
# RTX 50シリーズ（Blackwell sm_120）対応版
# CUDA 12.8 + PyTorch Nightlyビルドを使用
# VS Code Dev Container 専用設定
# ================================================

# CUDA 12.8 Ubuntu 22.04 ベースイメージ（Blackwell対応）
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 環境変数設定（sm_120対応）
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    TORCH_CUDA_ARCH_LIST="9.0;12.0" \
    CUDA_LAUNCH_BLOCKING=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    JUPYTER_TOKEN=research2025

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
    # VS Code Dev Container用の追加ツール
    sudo zsh \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11をデフォルトに設定
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Python仮想環境の作成
RUN python3.11 -m venv /opt/venv

# システムのデフォルトPythonを仮想環境のPythonに強制的に置き換える
RUN ln -sf /opt/venv/bin/python /usr/bin/python && \
    ln -sf /opt/venv/bin/python /usr/bin/python3 && \
    ln -sf /opt/venv/bin/pip /usr/bin/pip && \
    ln -sf /opt/venv/bin/pip /usr/bin/pip3

# 仮想環境のパスをPATHの先頭に追加
ENV PATH="/opt/venv/bin:$PATH"

# ===================================================
# RTX 50シリーズ対応: PyTorch Nightly (cu128) インストール
# ===================================================
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ===================================================
# Pythonパッケージ一括インストール
# ===================================================
RUN pip install --no-cache-dir six hatchling wheel

RUN pip install --no-cache-dir --no-build-isolation --timeout=6000 \
    # CuPy for CUDA 12.x（GPU4PySCF用）
    cupy-cuda12x==13.4.1 \
    cutensor-cu12==2.2.0 \
    \
    # 基本的な科学計算ライブラリ
    numpy==1.26.4 \
    scipy==1.13.0 \
    pandas==2.2.2 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    plotly==5.20.0 \
    jupyter==1.0.0 \
    jupyterlab==4.1.5 \
    ipython==8.23.0 \
    \
    # 機械学習フレームワーク
    tensorflow==2.16.1 \
    scikit-learn==1.4.2 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    catboost==1.2.3 \
    keras==3.1.1 \
    \
    # Deep Learning - Transformers
    transformers==4.40.0 \
    accelerate==0.29.3 \
    datasets==2.19.0 \
    tokenizers==0.19.1 \
    sentencepiece==0.2.0 \
    \
    # 計算化学ライブラリ
    rdkit==2024.03.1 \
    ase==3.22.1 \
    mdanalysis==2.7.0 \
    mdtraj==1.10.0 \
    pyscf==2.8.0 \
    geometric==1.1 \
    pubchempy==1.0.4 \
    py3Dmol==2.5.2 \
    openbabel-wheel==3.1.1.18 \
    chempy==0.8.3 \
    \
    # 分子モデリング
    biopython==1.79 \
    biotite==0.39.0 \
    git+https://github.com/prody/ProDy.git \
    oddt==0.7 \
    deepchem \
    \
    # データ処理とツール
    h5py==3.11.0 \
    netCDF4==1.6.5 \
    xarray==2024.3.0 \
    dask==2024.4.2 \
    bokeh==3.4.1 \
    altair==5.3.0 \
    networkx==3.3 \
    \
    # 開発ツール
    joblib==1.5.1 \
    tqdm==4.67.1 \
    rich==13.7.1 \
    pytest==8.1.1 \
    black==24.3.0 \
    flake8==7.0.0 \
    mypy==1.9.0 \
    pre-commit==3.7.0

# ===================================================
# GPU4PySCFをソースからビルド (sm_90, sm_120対応)
# ===================================================
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    git clone --depth 1 https://github.com/pyscf/gpu4pyscf.git /tmp/gpu4pyscf_build && \
    cd /tmp/gpu4pyscf_build && \
    CMAKE_CONFIGURE_ARGS="-DCUDA_ARCHITECTURES=90;120" pip install . && \
    cd / && \
    rm -rf /tmp/gpu4pyscf_build && \
    apt-get purge -y --auto-remove git && \
    rm -rf /var/lib/apt/lists/*

# Node.js最新化とCodex CLIのインストール
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest

# Codex CLIのインストール
RUN npm install -g @openai/codex

# PATHの問題を回避するため、実行ファイルを/usr/local/binにシンボリックリンクする
RUN ln -s /usr/bin/npx /usr/local/bin/npx

# 設定ディレクトリの作成
RUN mkdir -p /workspace/logs /root/.codex

# AGENTS.mdの作成
RUN cat <<'EOF' > /root/.codex/AGENTS.md
あなたは計算化学と機械学習に特化した強力なコーディングエージェントです。
あなたは科学技術計算のための包括的なPythonライブラリにアクセスできます。
あなたは以下のツールを使って、ファイル操作、コード編集、コマンド実行ができます:
- read_file: ファイルを読み込む
- write_file: ファイルを作成・書き込みする
- search_files: ファイルを検索する
- execute_shell_command: シェルコマンドを実行する
ユーザーの指示を正確に理解し、これらのツールを積極的に活用してタスクを実行してください。
必ず日本語で応答してください。
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

RUN chmod +x /usr/local/bin/verify-gpu.py

# 起動スクリプトを直接Dockerfileに埋め込む
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
SCRIPT

RUN chmod +x /usr/local/bin/start-environment.sh

# bashrcの設定: ターミナル起動時に自動で仮想環境を有効化
RUN echo "source /opt/venv/bin/activate" >> /root/.bashrc

# Pythonパス設定
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# ポート公開
EXPOSE 8888 9123

# エントリーポイント
CMD ["/usr/local/bin/start-environment.sh"]
