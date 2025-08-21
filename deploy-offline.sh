#!/bin/bash

# =========================================
# 計算化学・ML研究環境 オフラインデプロイスクリプト
# =========================================

set -e

# 色付き出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ===================
# 環境チェック
# ===================
check_requirements() {
    log_info "システム要件をチェックしています..."
    
    # Docker確認
    if ! command -v docker &> /dev/null; then
        log_error "Dockerがインストールされていません"
        exit 1
    fi
    
    # Docker Compose確認（新しいバージョン）
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
        log_success "Docker Compose V2が検出されました"
    elif command -v docker-compose &> /dev/null; then
        COMPOSE_CMD="docker-compose"
        log_success "Docker Compose V1が検出されました"
    else
        log_error "Docker Composeがインストールされていません"
        exit 1
    fi
    
    # NVIDIA Docker Runtime確認
    if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
        log_warning "NVIDIA Docker Runtimeが設定されていない可能性があります"
        read -p "続行しますか？ (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_success "NVIDIA Docker Runtime検出"
    fi
    
    log_success "システム要件チェック完了"
}

# ===================
# オンライン環境での準備
# ===================
prepare_online() {
    log_info "オンライン環境でイメージとデータを準備しています..."
    
    # プロジェクトディレクトリ作成
    PROJECT_DIR="computational-research-offline"
    mkdir -p $PROJECT_DIR
    cd $PROJECT_DIR
    
    # Dockerfileとdocker-compose.ymlをコピー
    log_info "設定ファイルをコピー中..."
    # ここでDockerfileとdocker-compose.ymlが現在のディレクトリにあることを前提
    
    # イメージのビルド
    log_info "Dockerイメージをビルド中（時間がかかります）..."
    $COMPOSE_CMD build --no-cache
    
    # イメージの保存
    log_info "イメージをファイルに保存中..."
    docker save computational-chemistry-ml:latest | gzip > comp-chem-ml-image.tar.gz
    
    # Ollamaモデルのダウンロード
    log_info "Ollamaモデルをダウンロード中..."
    if command -v ollama &> /dev/null; then
        ollama pull qwen2.5-coder:7b-instruct
        ollama pull deepseek-coder:33b-instruct
        ollama pull codellama:13b
        
        # Ollamaモデルのエクスポート
        log_info "Ollamaモデルをエクスポート中..."
        mkdir -p ollama-models
        for model in qwen2.5-coder:7b-instruct deepseek-coder:33b-instruct codellama:13b; do
            model_name=$(echo $model | sed 's/:/-/g')
            ollama show $model --modelfile > ollama-models/${model_name}.modelfile
        done
    else
        log_warning "Ollamaがインストールされていません。手動でモデルを準備してください"
    fi
    
    # 必要なPythonパッケージのダウンロード（オプション）
    log_info "追加のPythonパッケージをダウンロード中..."
    mkdir -p python-packages
    docker run --rm -v $(pwd)/python-packages:/packages computational-chemistry-ml:latest \
        pip download -d /packages \
        rdkit ase mdanalysis pyscf torch torchvision
    
    # アーカイブの作成
    log_info "デプロイメントパッケージを作成中..."
    tar -czf computational-research-deployment.tar.gz \
        comp-chem-ml-image.tar.gz \
        docker-compose.yml \
        Dockerfile \
        requirements.txt \
        ollama-models/ \
        python-packages/ \
        *.md \
        *.sh
    
    log_success "オンライン準備完了！"
    log_info "computational-research-deployment.tar.gz をオフライン環境に転送してください"
}

# ===================
# オフライン環境でのデプロイ
# ===================
deploy_offline() {
    log_info "オフライン環境でデプロイを開始します..."
    
    # アーカイブの展開
    if [ -f "computational-research-deployment.tar.gz" ]; then
        log_info "デプロイメントパッケージを展開中..."
        tar -xzf computational-research-deployment.tar.gz
    else
        log_error "デプロイメントパッケージが見つかりません"
        exit 1
    fi
    
    # Dockerイメージのロード
    log_info "Dockerイメージをロード中..."
    docker load < comp-chem-ml-image.tar.gz
    
    # ディレクトリ構造の作成
    log_info "作業ディレクトリを作成中..."
    mkdir -p workspace/{notebooks,scripts,data}
    mkdir -p config/{claude,serena,claude-bridge}
    mkdir -p datasets models logs notebooks
    
    # Ollamaのセットアップ（手動）
    log_warning "Ollamaモデルは手動でインポートしてください："
    echo "  ollama create qwen2.5-coder:7b-instruct -f ollama-models/qwen2.5-coder-7b-instruct.modelfile"
    
    # コンテナの起動
    log_info "Dockerコンテナを起動中..."
    $COMPOSE_CMD up -d
    
    # ヘルスチェック
    log_info "サービスの起動を確認中..."
    sleep 10
    
    if curl -s http://localhost:8888 > /dev/null; then
        log_success "JupyterLabが起動しました"
    else
        log_warning "JupyterLabの起動を確認できません"
    fi
    
    log_success "デプロイ完了！"
    show_access_info
}

# ===================
# アクセス情報表示
# ===================
show_access_info() {
    echo
    echo "=========================================="
    echo "  計算化学・ML研究環境 アクセス情報"
    echo "=========================================="
    echo
    echo "📊 JupyterLab:"
    echo "   URL: http://localhost:8888"
    echo "   Token: research2025"
    echo
    echo "🤖 Claude Code:"
    echo "   docker exec -it comp-chem-ml-env claude"
    echo
    echo "🌉 Claude-bridge API:"
    echo "   URL: http://localhost:8080"
    echo
    echo "🎯 Serena-MCP:"
    echo "   URL: http://localhost:9121"
    echo "   Dashboard: http://localhost:9122"
    echo
    echo "📁 作業ディレクトリ:"
    echo "   Host: $(pwd)/workspace"
    echo "   Container: /workspace"
    echo
    echo "🔧 CUDA Version: 12.4.1"
    echo "🐍 PyTorch Version: 2.5.1"
    echo
    echo "=========================================="
}

# ===================
# クリーンアップ
# ===================
cleanup() {
    log_info "環境をクリーンアップしています..."
    
    read -p "すべてのコンテナとデータを削除しますか？ (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        $COMPOSE_CMD down -v
        rm -rf workspace config datasets models logs notebooks
        log_success "クリーンアップ完了"
    else
        log_info "クリーンアップをキャンセルしました"
    fi
}

# ===================
# メインメニュー
# ===================
show_menu() {
    echo
    echo "=========================================="
    echo "  計算化学・ML研究環境 デプロイツール"
    echo "=========================================="
    echo
    echo "1) システム要件チェック"
    echo "2) オンライン環境で準備（イメージビルド）"
    echo "3) オフライン環境にデプロイ"
    echo "4) アクセス情報を表示"
    echo "5) 環境をクリーンアップ"
    echo "6) 終了"
    echo
    read -p "選択してください [1-6]: " choice
    
    case $choice in
        1)
            check_requirements
            ;;
        2)
            check_requirements
            prepare_online
            ;;
        3)
            check_requirements
            deploy_offline
            ;;
        4)
            show_access_info
            ;;
        5)
            cleanup
            ;;
        6)
            log_info "終了します"
            exit 0
            ;;
        *)
            log_error "無効な選択です"
            ;;
    esac
}

# ===================
# メイン処理
# ===================
main() {
    # 引数がある場合は直接実行
    if [ $# -gt 0 ]; then
        case $1 in
            check)
                check_requirements
                ;;
            prepare)
                check_requirements
                prepare_online
                ;;
            deploy)
                check_requirements
                deploy_offline
                ;;
            clean)
                cleanup
                ;;
            *)
                log_error "無効なコマンド: $1"
                echo "使用方法: $0 [check|prepare|deploy|clean]"
                exit 1
                ;;
        esac
    else
        # 引数がない場合はメニュー表示
        while true; do
            show_menu
        done
    fi
}

# スクリプト実行
main "$@"
