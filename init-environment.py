#!/usr/bin/env python3
"""
計算化学・機械学習研究環境 初期化スクリプト
環境の動作確認と初期設定を行います
"""

import os
import sys
import json
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

# カラー出力用
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def print_colored(message: str, color: str = Colors.WHITE):
    """色付きメッセージを出力"""
    print(f"{color}{message}{Colors.RESET}")

def print_header(title: str):
    """ヘッダーを出力"""
    print()
    print_colored("=" * 60, Colors.CYAN)
    print_colored(f"  {title}", Colors.CYAN)
    print_colored("=" * 60, Colors.CYAN)
    print()

def check_gpu() -> bool:
    """GPU環境をチェック"""
    print_header("🎮 GPU環境チェック")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print_colored(f"✅ CUDA利用可能", Colors.GREEN)
            print(f"  - デバイス数: {device_count}")
            print(f"  - GPU名: {device_name}")
            print(f"  - メモリ: {memory:.1f} GB")
            
            # 簡単なテスト
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor)
            print_colored("✅ GPU演算テスト成功", Colors.GREEN)
            
            return True
        else:
            print_colored("⚠️ CUDAが利用できません", Colors.YELLOW)
            return False
            
    except ImportError:
        print_colored("❌ PyTorchがインストールされていません", Colors.RED)
        return False
    except Exception as e:
        print_colored(f"❌ エラー: {e}", Colors.RED)
        return False

def check_libraries() -> Dict[str, bool]:
    """必要なライブラリの存在をチェック"""
    print_header("📚 ライブラリチェック")
    
    libraries = {
        # 計算化学
        "rdkit": "RDKit (ケモインフォマティクス)",
        "ase": "ASE (原子シミュレーション)",
        "MDAnalysis": "MDAnalysis (分子動力学)",
        "pyscf": "PySCF (量子化学)",
        
        # 機械学習
        "torch": "PyTorch",
        "tensorflow": "TensorFlow",
        "sklearn": "scikit-learn",
        "xgboost": "XGBoost",
        "transformers": "Transformers",
        
        # データサイエンス
        "numpy": "NumPy",
        "scipy": "SciPy",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "jupyter": "Jupyter",
    }
    
    results = {}
    
    for lib, name in libraries.items():
        try:
            importlib.import_module(lib)
            print_colored(f"✅ {name}: インストール済み", Colors.GREEN)
            results[lib] = True
        except ImportError:
            print_colored(f"❌ {name}: 未インストール", Colors.RED)
            results[lib] = False
    
    # サマリー
    installed = sum(results.values())
    total = len(results)
    
    print()
    if installed == total:
        print_colored(f"🎉 すべてのライブラリが利用可能です ({installed}/{total})", Colors.GREEN)
    else:
        print_colored(f"⚠️ 一部のライブラリが不足しています ({installed}/{total})", Colors.YELLOW)
    
    return results

def check_ollama() -> bool:
    """Ollamaの接続をチェック"""
    print_header("🤖 Ollama接続チェック")
    
    try:
        import requests
        
        # Ollamaサーバーの確認
        response = requests.get("http://host.docker.internal:11434/api/tags", timeout=5)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            print_colored("✅ Ollamaサーバーに接続成功", Colors.GREEN)
            
            if models:
                print("  利用可能なモデル:")
                for model in models:
                    name = model.get("name", "unknown")
                    size = model.get("size", 0) / 1e9
                    print(f"    - {name} ({size:.1f} GB)")
            else:
                print_colored("  ⚠️ モデルがインストールされていません", Colors.YELLOW)
            
            return True
        else:
            print_colored("❌ Ollamaサーバーへの接続に失敗", Colors.RED)
            return False
            
    except requests.exceptions.ConnectionError:
        print_colored("❌ Ollamaサーバーが起動していません", Colors.RED)
        print("  ホスト側で以下を実行してください:")
        print("    ollama serve")
        return False
    except Exception as e:
        print_colored(f"❌ エラー: {e}", Colors.RED)
        return False

def check_services() -> Dict[str, bool]:
    """各サービスの状態をチェック"""
    print_header("🔧 サービス状態チェック")
    
    services = {
        "http://localhost:8888": "JupyterLab",
        "http://localhost:8080": "Claude-bridge",
        "http://localhost:9121": "Serena-MCP",
        "http://localhost:9122": "Serena Dashboard",
    }
    
    results = {}
    
    try:
        import requests
        
        for url, name in services.items():
            try:
                response = requests.get(url, timeout=3)
                if response.status_code < 500:
                    print_colored(f"✅ {name}: 稼働中 ({url})", Colors.GREEN)
                    results[name] = True
                else:
                    print_colored(f"⚠️ {name}: エラー応答 ({url})", Colors.YELLOW)
                    results[name] = False
            except:
                print_colored(f"❌ {name}: 接続不可 ({url})", Colors.RED)
                results[name] = False
                
    except ImportError:
        print_colored("❌ requestsライブラリが必要です", Colors.RED)
        
    return results

def create_sample_notebook():
    """サンプルノートブックを作成"""
    print_header("📓 サンプルノートブック作成")
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 計算化学・機械学習研究環境 サンプルノートブック\\n",
                    "\\n",
                    "このノートブックでは、環境の基本的な使い方を示します。"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 必要なライブラリのインポート\\n",
                    "import numpy as np\\n",
                    "import pandas as pd\\n",
                    "import matplotlib.pyplot as plt\\n",
                    "from rdkit import Chem\\n",
                    "from rdkit.Chem import Descriptors\\n",
                    "import torch\\n",
                    "\\n",
                    "print('環境の準備完了！')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. RDKitを使った分子操作"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# アスピリンの分子を作成\\n",
                    "aspirin = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')\\n",
                    "\\n",
                    "# 分子特性の計算\\n",
                    "mw = Descriptors.MolWt(aspirin)\\n",
                    "logp = Descriptors.MolLogP(aspirin)\\n",
                    "hbd = Descriptors.NumHDonors(aspirin)\\n",
                    "hba = Descriptors.NumHAcceptors(aspirin)\\n",
                    "\\n",
                    "print(f'分子量: {mw:.2f}')\\n",
                    "print(f'LogP: {logp:.2f}')\\n",
                    "print(f'水素結合ドナー: {hbd}')\\n",
                    "print(f'水素結合アクセプター: {hba}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. PyTorchでの簡単な機械学習"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# GPUの確認\\n",
                    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\\n",
                    "print(f'使用デバイス: {device}')\\n",
                    "\\n",
                    "# 簡単なニューラルネットワーク\\n",
                    "class SimpleNet(torch.nn.Module):\\n",
                    "    def __init__(self):\\n",
                    "        super().__init__()\\n",
                    "        self.fc1 = torch.nn.Linear(10, 50)\\n",
                    "        self.fc2 = torch.nn.Linear(50, 1)\\n",
                    "    \\n",
                    "    def forward(self, x):\\n",
                    "        x = torch.relu(self.fc1(x))\\n",
                    "        return self.fc2(x)\\n",
                    "\\n",
                    "model = SimpleNet().to(device)\\n",
                    "print(f'モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    # ノートブックを保存
    notebook_path = Path("/workspace/notebooks/sample_notebook.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(notebook_path, "w") as f:
        json.dump(notebook_content, f, indent=2)
    
    print_colored(f"✅ サンプルノートブックを作成しました: {notebook_path}", Colors.GREEN)

def main():
    """メイン処理"""
    print_colored("""
    ╔══════════════════════════════════════════════════════════╗
    ║   計算化学・機械学習研究環境 初期化スクリプト           ║
    ║   Computational Chemistry & ML Research Environment     ║
    ╚══════════════════════════════════════════════════════════╝
    """, Colors.MAGENTA)
    
    # 各種チェック
    gpu_ok = check_gpu()
    libraries = check_libraries()
    ollama_ok = check_ollama()
    services = check_services()
    
    # サンプル作成
    try:
        create_sample_notebook()
    except Exception as e:
        print_colored(f"サンプルノートブックの作成に失敗: {e}", Colors.YELLOW)
    
    # 最終レポート
    print_header("📋 環境診断レポート")
    
    all_ok = True
    
    if gpu_ok:
        print_colored("✅ GPU: 正常", Colors.GREEN)
    else:
        print_colored("⚠️ GPU: 問題あり", Colors.YELLOW)
        all_ok = False
    
    if all(libraries.values()):
        print_colored("✅ ライブラリ: すべて利用可能", Colors.GREEN)
    else:
        missing = [k for k, v in libraries.items() if not v]
        print_colored(f"⚠️ ライブラリ: {len(missing)}個不足", Colors.YELLOW)
        all_ok = False
    
    if ollama_ok:
        print_colored("✅ Ollama: 接続成功", Colors.GREEN)
    else:
        print_colored("⚠️ Ollama: 接続失敗", Colors.YELLOW)
        all_ok = False
    
    if all(services.values()):
        print_colored("✅ サービス: すべて稼働中", Colors.GREEN)
    else:
        print_colored("⚠️ サービス: 一部停止中", Colors.YELLOW)
        all_ok = False
    
    print()
    if all_ok:
        print_colored("🎉 環境は完全に動作しています！", Colors.GREEN)
        print_colored("   JupyterLabにアクセス: http://localhost:8888", Colors.CYAN)
        print_colored("   Token: research2025", Colors.CYAN)
    else:
        print_colored("⚠️ 一部の機能に問題があります。ログを確認してください。", Colors.YELLOW)
    
    print()
    print_colored("Happy Research! 🧪🤖💻", Colors.MAGENTA)

if __name__ == "__main__":
    main()