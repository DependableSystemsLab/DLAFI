cd /workspace

pip install pyyaml
# ========= BUILDING LLVM =========
echo "========= BUILDING LLVM ========="

git clone https://github.com/llvm/llvm-project.git
cd llvm-project && git checkout 9778ec057cf4 && cd ..
mkdir llvm-project/build 
cd llvm-project/build

cmake -G Ninja ../llvm \
  	-DLLVM_ENABLE_PROJECTS="clang;mlir" \
  	-DLLVM_BUILD_TESTS=ON \
  	-DLLVM_TARGETS_TO_BUILD="host" \
  	-DLLVM_ENABLE_ASSERTIONS=ON \
  	-DLLVM_ENABLE_RTTI=ON


cmake --build . --target clang check-mlir mlir-translate opt llc lli llvm-dis llvm-link -j 16

ninja install -j 16

# ========= BUILDING ONNX MLIR =========
echo " ========= BUILDING ONNX MLIR ========="

cd /workspace

git clone --recursive https://github.com/DependableSystemsLab/onnx-mlir-lltfi.git
mv onnx-mlir-lltfi onnx-mlir && cd onnx-mlir
git checkout LLTFI
cd ..
mkdir onnx-mlir/build && cd onnx-mlir/build

cmake -G Ninja \
	-DCMAKE_CXX_COMPILER=/usr/bin/c++ \
	-DMLIR_DIR=${MLIR_DIR} \
	.. 

cmake --build .

export LIT_OPTS=-v
cmake --build . --target check-onnx-lit

ninja install

# ========= BUILDING DLAFI =========
echo "========= BUILDING DLAFI ========="

cd /workspace/DLAFI/llfi-dlafi
./setup -LLFI_BUILD_ROOT /workspace/llfi-build -LLVM_SRC_ROOT /workspace/llvm-project/ -LLVM_DST_ROOT /workspace/llvm-project/build/ 

cd /workspace/DLAFI/pytorch-fi
apt update && apt install -y python3-venv
python3 -m venv pytorch-env

source pytorch-env/bin/activate

python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
