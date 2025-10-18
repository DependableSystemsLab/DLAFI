#!/bin/bash
set -e  # Exit immediately if a command fails
set -o pipefail

cd /workspace

# ========= PYTHON DEPENDENCIES =========
echo "========= Installing Python dependencies ========="
pip install --no-cache-dir pyyaml

# ========= BUILDING LLVM =========
echo "========= BUILDING LLVM ========="

git clone https://github.com/llvm/llvm-project.git "$LLVM_ROOT"
cd "$LLVM_ROOT"
git checkout 9778ec057cf4
cd /workspace
mkdir -p "$LLVM_DST_ROOT"
cd "$LLVM_DST_ROOT"

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_BUILD_TESTS=ON \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON

cmake --build . --target clang check-mlir mlir-translate opt llc lli llvm-dis llvm-link -j $(nproc)
ninja install -j $(nproc)

# ========= BUILDING ONNX-MLIR =========
echo "========= BUILDING ONNX-MLIR ========="

cd /workspace
git clone --recursive https://github.com/DependableSystemsLab/onnx-mlir-lltfi.git "$ONNX_MLIR_ROOT"
cd "$ONNX_MLIR_ROOT"
git checkout LLTFI

mkdir -p build && cd build
cmake -G Ninja \
  -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
  -DMLIR_DIR="${MLIR_DIR}" \
  ..

cmake --build . -j $(nproc)

export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
ninja install

# ========= BUILDING DLAFI =========
echo "========= BUILDING DLAFI ========="

cd "$DLAFI_ROOT/llfi-dlafi"
./setup \
  -LLFI_BUILD_ROOT "$LLFI_BUILD_ROOT" \
  -LLVM_SRC_ROOT "$LLVM_ROOT" \
  -LLVM_DST_ROOT "$LLVM_DST_ROOT"

# ========= SETTING UP PYTORCH-FI =========
echo "========= SETTING UP PYTORCH-FI ========="

cd "$DLAFI_ROOT/pytorch-fi"
apt-get update && apt-get install -y python3-venv

python3 -m venv pytorch-env
source pytorch-env/bin/activate

python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "========= DLAFI INSTALLATION COMPLETE ========="
