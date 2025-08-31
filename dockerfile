# ================================================
# RTX 50シリーズ（Blackwell sm_120）対応版
# CUDA 12.8 + PyTorch Nightlyビルドを使用
# Codex CLI と Ollama を連携
# ================================================

# CUDA 12.8 Ubuntu 22.04 ベースイメージ（Blackwell対応）
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

# 環境変数設定（sm_120対応 + Ollama連携）
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_API_BASE=http://host.docker.internal:11434/v1 \
    OLLAMA_MODEL=gpt-oss:20b \
    OPENAI_API_KEY=dummy \
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

# Python仮想環境の作成と有効化
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ===================================================
# RTX 50シリーズ対応: PyTorch Nightly (cu128) インストール
# ===================================================
RUN pip install --no-cache-dir --pre \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ===================================================
# Pythonパッケージ一括インストール
# 依存関係の問題を回避するため、ビルドツール(six, hatchling)を先にインストール
# ===================================================
RUN pip install --no-cache-dir six hatchling wheel

RUN pip install --no-cache-dir --no-build-isolation \
    # CuPy for CUDA 12.x（GPU4PySCF用）
    cupy-cuda12x==13.6.0 \
    cutensor-cu12 \
    \
    # 基本的な科学計算ライブラリ (numpyバージョンを固定)
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
    # 機械学習フレームワーク（PyTorch以外）
    tensorflow==2.16.1 \
    scikit-learn==1.4.2 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    catboost==1.2.3 \
    keras==3.1.1 \
    \
    # Deep Learning - Transformers & Ecosystem
    transformers==4.40.0 \
    accelerate==0.29.3 \
    datasets==2.19.0 \
    tokenizers==0.19.1 \
    sentencepiece==0.2.0 \
    \
    # 計算化学ライブラリ（GPU対応版含む）
    rdkit==2024.03.1 \
    ase==3.22.1 \
    mdanalysis==2.7.0 \
    mdtraj==1.10.0 \
    pyscf==2.8.0 \
    gpu4pyscf-cuda12x==1.4.2 \
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
    pre-commit==3.7.0 \
    \
    # Serena-MCP
    git+https://github.com/oraios/serena.git

# Node.js最新化とCodex CLIのインストール
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest

# Codex CLIのインストール
RUN npm install -g @openai/codex

# PATHの問題を回避するため、実行ファイルを/usr/local/binにシンボリックリンクする
RUN ln -s /opt/venv/bin/serena /usr/local/bin/serena && \
    ln -s /usr/bin/npx /usr/local/bin/npx

# 設定ディレクトリの作成
RUN mkdir -p /workspace/logs /root/.codex /root/.serena

# Serena-MCP設定
RUN cat <<EOF > /root/.serena/serena_config.yml
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

# Codex CLIのインストール
RUN npm install -g @openai/codex

# --- MCPサーバー用のラッパースクリプトを作成 ---
# これにより、codexからの呼び出し時に環境変数の問題を確実に回避する

# 1. Serena用ラッパースクリプト
RUN echo '#!/bin/sh' > /usr/local/bin/run-serena-mcp.sh && \
    echo 'exec /opt/venv/bin/serena start-mcp-server --context agent --transport stdio' >> /usr/local/bin/run-serena-mcp.sh

# 2. Filesystem用ラッパースクリプト
RUN echo '#!/bin/sh' > /usr/local/bin/run-filesystem-mcp.sh && \
    echo 'exec /usr/bin/npx -y @modelcontextprotocol/server-filesystem /workspace' >> /usr/local/bin/run-filesystem-mcp.sh

# 3. スクリプトに実行権限を付与
RUN chmod +x /usr/local/bin/run-serena-mcp.sh && \
    chmod +x /usr/local/bin/run-filesystem-mcp.sh

# 設定ディレクトリの作成
RUN mkdir -p /workspace/logs /root/.codex /root/.serena

# ... (serena_config.yml の設定は変更なし) ...

# Codex CLI設定 (ラッパースクリプトを呼び出す方式)
RUN cat <<'EOF' > /root/.codex/config.toml
# Default model provider to use.
model_provider = "ollama"

# Default model to use.
model = "gpt-oss:20b"

[model_providers.ollama]
name = "Ollama"
base_url = "http://ollama:11434/v1"
api_key_env = ""

[mcp_servers]
# Serena-MCP: ラッパースクリプト経由で起動
[mcp_servers.serena]
command = "/usr/local/bin/run-serena-mcp.sh"

# Filesystem-MCP: ラッパースクリプト経由で起動
[mcp_servers.filesystem]
command = "/usr/local/bin/run-filesystem-mcp.sh"
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

# 起動スクリプトをコピー（外部ファイルから）
COPY start-environment.sh /usr/local/bin/start-environment.sh
RUN chmod +x /usr/local/bin/start-environment.sh

# Pythonパス設定
ENV PYTHONPATH="/workspace:$PYTHONPATH"

# ポート公開
EXPOSE 8888 9121 9122 9123

# ボリュームマウントポイント
VOLUME ["/workspace", "/root/.codex", "/root/.serena", "/workspace/logs"]

# エントリーポイント
ENTRYPOINT ["/usr/local/bin/start-environment.sh"]
