printf "\n[Compile Script]: Getting the ONNX model\n"
FILE=model.onnx
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
    echo "$FILE does not exist."
    source $DLAFI_ROOT/pytorch-fi/pytorch-env/bin/activate
    python3 -m transformers.onnx --model=microsoft/codebert-base-mlm --feature=masked-lm  --opset 14 .
    deactivate

fi

printf "\n[Compile Script]: Convert TF model to LLVM IR\n"
onnx-mlir --EmitLLVMIR  --instrument-onnx-ops="ALL" --InstrumentBeforeAndAfterOp model.onnx
mlir-translate -mlir-to-llvmir model.onnx.mlir > model.mlir.ll

printf "\n[Compile Script]: Compile main driver program and link to TF model in LLVM IR\n"
clang++ -DONNX_ML=1 input.c -o main.ll -O0 -S -emit-llvm -lonnx_proto -lprotobuf -I$ONNX_MLIR_SRC/include
llvm-link -o model.ll -S main.ll model.mlir.ll

printf "\n[Compile Script]: Generate model.exe \n"
llc -filetype=obj -o model.o model.ll -O0 --relocation-model=pic
clang++ -o model.exe model.o -L$LLFI_BUILD_ROOT/bin/../runtime_lib -lllfi-rt -lpthread -L /Debug/lib -Wl,-rpath $LLFI_BUILD_ROOT/bin/../runtime_lib -I$ONNX_MLIR_SRC/include -O0 -lonnx_proto -lprotobuf -lcruntime -ljson-c

printf "\n[Compile Script]: Compilation complete\n"