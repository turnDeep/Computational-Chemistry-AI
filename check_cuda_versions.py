#!/usr/bin/env python3
"""
CUDAドライバとランタイムのバージョン不一致チェックツール
============================================

このスクリプトは以下を確認します：
1. CUDAドライババージョン (nvidia-smiから取得)
2. CUDAランタイムバージョン (nvccから取得)
3. PyTorchが認識しているCUDAバージョン
4. CuPyが認識しているCUDAバージョン
5. これらのバージョン間の互換性

Usage:
    python3 check_cuda_versions.py
"""

import subprocess
import sys
import os
import re
from typing import Optional, Tuple


def get_driver_version() -> Optional[str]:
    """nvidia-smiからCUDAドライババージョンを取得"""
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # CUDA Version: 12.8 のような行を探す
            match = re.search(r'CUDA Version:\s+(\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)

            # より詳細なドライババージョンを探す
            match = re.search(r'Driver Version:\s+(\d+\.\d+\.\d+)', result.stdout)
            if match:
                driver_ver = match.group(1)
                return f"{driver_ver}"
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_runtime_version() -> Optional[str]:
    """nvccからCUDAランタイムバージョンを取得"""
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # release 12.8, V12.8.xxx のような行を探す
            match = re.search(r'release\s+(\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def get_pytorch_cuda_version() -> Tuple[Optional[str], Optional[str]]:
    """PyTorchが認識しているCUDAバージョンとビルド情報を取得"""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if not cuda_available:
            return None, "CUDA not available"

        # PyTorchがビルドされたCUDAバージョン
        cuda_version = torch.version.cuda

        # PyTorchバージョン
        pytorch_version = torch.__version__

        return cuda_version, pytorch_version
    except ImportError:
        return None, "PyTorch not installed"


def get_cupy_cuda_version() -> Optional[str]:
    """CuPyが認識しているCUDAバージョンを取得"""
    try:
        import cupy as cp

        # CuPyのCUDAランタイムバージョン
        runtime_version = cp.cuda.runtime.runtimeGetVersion()
        major = runtime_version // 1000
        minor = (runtime_version % 1000) // 10

        return f"{major}.{minor}"
    except ImportError:
        return "CuPy not installed"
    except Exception as e:
        return f"Error: {str(e)}"


def get_cuda_home() -> Optional[str]:
    """CUDA_HOME環境変数を取得"""
    return os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')


def check_version_compatibility(driver_ver: str, runtime_ver: str) -> Tuple[bool, str]:
    """
    CUDAドライバとランタイムの互換性をチェック

    ルール：
    - ドライバのCUDAバージョン >= ランタイムのCUDAバージョン であること
    - ドライバは後方互換性がある
    """
    try:
        driver_major, driver_minor = map(int, driver_ver.split('.')[:2])
        runtime_major, runtime_minor = map(int, runtime_ver.split('.')[:2])

        if driver_major > runtime_major:
            return True, "✅ 互換性あり（ドライバが新しい）"
        elif driver_major == runtime_major:
            if driver_minor >= runtime_minor:
                return True, "✅ 互換性あり"
            else:
                return False, f"⚠️ 警告: ドライバ({driver_ver})がランタイム({runtime_ver})より古い可能性"
        else:
            return False, f"❌ 非互換: ドライバ({driver_ver})がランタイム({runtime_ver})より古い"
    except ValueError:
        return False, "⚠️ バージョン解析エラー"


def main():
    print("=" * 70)
    print("CUDAドライバ・ランタイム バージョン確認ツール")
    print("=" * 70)
    print()

    # 1. CUDAドライババージョン
    print("📌 1. CUDAドライババージョン (nvidia-smi)")
    print("-" * 70)
    driver_version = get_driver_version()
    if driver_version:
        print(f"   CUDAドライババージョン: {driver_version}")
    else:
        print("   ❌ nvidia-smiが利用できません（GPUドライバが正しくインストールされていない可能性）")
    print()

    # 2. CUDAランタイムバージョン
    print("📌 2. CUDAランタイムバージョン (nvcc)")
    print("-" * 70)
    runtime_version = get_runtime_version()
    if runtime_version:
        print(f"   CUDAランタイムバージョン: {runtime_version}")
    else:
        print("   ⚠️ nvccが利用できません（CUDA Toolkitがインストールされていない可能性）")

    cuda_home = get_cuda_home()
    if cuda_home:
        print(f"   CUDA_HOME: {cuda_home}")
    else:
        print("   ⚠️ CUDA_HOME環境変数が設定されていません")
    print()

    # 3. PyTorchのCUDAバージョン
    print("📌 3. PyTorchが認識しているCUDAバージョン")
    print("-" * 70)
    pytorch_cuda_ver, pytorch_ver = get_pytorch_cuda_version()
    if pytorch_cuda_ver:
        print(f"   PyTorchバージョン: {pytorch_ver}")
        print(f"   PyTorchビルド時のCUDAバージョン: {pytorch_cuda_ver}")
        print(f"   CUDA利用可能: {'✅ Yes' if pytorch_cuda_ver else '❌ No'}")
    else:
        print(f"   ⚠️ {pytorch_ver}")
    print()

    # 4. CuPyのCUDAバージョン
    print("📌 4. CuPyが認識しているCUDAバージョン")
    print("-" * 70)
    cupy_cuda_ver = get_cupy_cuda_version()
    print(f"   CuPy CUDAランタイムバージョン: {cupy_cuda_ver}")
    print()

    # 5. 互換性チェック
    print("📌 5. バージョン互換性チェック")
    print("-" * 70)

    all_compatible = True

    # ドライバとランタイムの互換性
    if driver_version and runtime_version:
        compatible, message = check_version_compatibility(driver_version, runtime_version)
        print(f"   ドライバ vs ランタイム: {message}")
        all_compatible = all_compatible and compatible
    else:
        print("   ⚠️ ドライバまたはランタイムが検出できないため、互換性チェックができません")
        all_compatible = False

    # PyTorchとランタイムの互換性
    if pytorch_cuda_ver and runtime_version:
        if pytorch_cuda_ver == runtime_version:
            print(f"   PyTorch vs ランタイム: ✅ 一致 ({pytorch_cuda_ver})")
        else:
            print(f"   PyTorch vs ランタイム: ⚠️ 不一致 (PyTorch: {pytorch_cuda_ver}, ランタイム: {runtime_version})")
            print(f"      → PyTorchは異なるCUDAバージョンでビルドされています")
            all_compatible = False

    # CuPyとランタイムの互換性
    if cupy_cuda_ver and "Error" not in cupy_cuda_ver and "not installed" not in cupy_cuda_ver:
        if runtime_version and cupy_cuda_ver == runtime_version:
            print(f"   CuPy vs ランタイム: ✅ 一致 ({cupy_cuda_ver})")
        elif runtime_version:
            print(f"   CuPy vs ランタイム: ⚠️ 不一致 (CuPy: {cupy_cuda_ver}, ランタイム: {runtime_version})")

    print()
    print("=" * 70)

    if all_compatible:
        print("✅ 全てのCUDAバージョンは互換性があります")
        print("=" * 70)
        return 0
    else:
        print("⚠️ バージョン不一致または互換性の問題が検出されました")
        print()
        print("【推奨される対処法】")
        print("1. NVIDIAドライバを最新版にアップデート")
        print("2. PyTorchを正しいCUDAバージョンで再インストール:")
        if runtime_version:
            print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{runtime_version.replace('.', '')}")
        print("3. 環境変数CUDA_HOMEが正しく設定されているか確認")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
