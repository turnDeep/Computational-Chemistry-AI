#!/bin/bash
set -e

echo "🔄 Reinstalling gpu4pyscf with LibXC enabled..."

# Uninstall existing version
pip uninstall -y gpu4pyscf

# Clone and install with correct flags
git clone --depth 1 https://github.com/pyscf/gpu4pyscf.git /tmp/gpu4pyscf_fix
cd /tmp/gpu4pyscf_fix

# Install with LibXC enabled (removing -DBUILD_LIBXC=OFF) and targeting sm_90/sm_120
export CMAKE_CONFIGURE_ARGS="-DCUDA_ARCHITECTURES=90;120"
pip install .

# Cleanup
cd /
rm -rf /tmp/gpu4pyscf_fix

echo "✅ gpu4pyscf reinstallation complete!"
