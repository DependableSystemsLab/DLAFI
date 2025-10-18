#!/bin/bash
set -e
set -o pipefail

cd /workspace
pip install --no-cache-dir pyyaml

# ========= BUILDING LLVM =========
echo "========= BUILDING LLVM ========="

export LLVM_SRC=/workspace/llvm-project
export LLVM_DST_ROOT=$LLVM_SRC/build

git clone https://github.com/llvm/llvm-project.git "$LLVM_SRC"
cd "$LLVM_SRC" && git checkout 9778ec057cf4 && cd ..
mkdir -p "$LLVM_DST_ROOT"
cd "$LLVM_DST_ROOT"

cmake -G Ninja ../llvm \
    -DLLVM_ENABLE_PROJECTS="clang;mlir" \
    -DLLVM_BUILD_TESTS=ON \
    -DLLVM_TARGETS_TO_BUILD="host" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON

cmake --build . --target clang check-mlir mlir-translate opt llc lli llvm-dis llvm-link -j"$(nproc)"
ninja install -j"$(nproc)"

# ========= INSTALLING PROTOBUF =========
echo "========= INSTALLING PROTOBUF v3.17.2 ========="
cd /workspace

curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v3.17.2/protobuf-all-3.17.2.zip
unzip -q protobuf-all-3.17.2.zip
cd protobuf-3.17.2

./configure
make -j"$(nproc)"
# make check   # Optional test phase
make install
ldconfig  # Refresh shared library cache

# Cleanup to save space
cd /workspace
rm -rf protobuf-3.17.2 protobuf-all-3.17.2.zip
echo "Protobuf v3.17.2 installation complete."

# ========= BUILDING ONNX-MLIR =========
echo "========= BUILDING ONNX-MLIR ========="

export ONNX_MLIR_SRC=/workspace/onnx-mlir
export ONNX_MLIR_BUILD=$ONNX_MLIR_SRC/build

git clone --recursive https://github.com/DependableSystemsLab/onnx-mlir-lltfi.git "$ONNX_MLIR_SRC"
cd "$ONNX_MLIR_SRC" && git checkout LLTFI && cd ..

mkdir -p "$ONNX_MLIR_BUILD"
cd "$ONNX_MLIR_BUILD"

cmake -G Ninja \
    -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
    -DMLIR_DIR=${LLVM_DST_ROOT}/lib/cmake/mlir \
    ..

cmake --build . -j"$(nproc)"
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
ninja install

# ========= BUILDING DLAFI =========
echo "========= BUILDING DLAFI ========="

export LLFI_BUILD_ROOT=/workspace/llfi-build
export DLAFI_ROOT=/workspace/DLAFI

cd "$DLAFI_ROOT/llfi-dlafi"
./setup \
    -LLFI_BUILD_ROOT "$LLFI_BUILD_ROOT" \
    -LLVM_SRC_ROOT "$LLVM_SRC" \
    -LLVM_DST_ROOT "$LLVM_DST_ROOT"

#setup JSON-C for LLFI tools if needed
bash "$DLAFI_ROOT/llfi-dlafi/tools/json-c-setup.sh"

cd "$DLAFI_ROOT/pytorch-fi"
apt-get update && apt-get install -y python3-venv

python3 -m venv pytorch-env
source pytorch-env/bin/activate

python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "========= DLAFI INSTALLATION COMPLETE ========="
