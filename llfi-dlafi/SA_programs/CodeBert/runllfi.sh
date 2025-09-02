rm -rf llfi*
mkdir FIutil

rm -f input*_*.pb
source $DLAFI_ROOT/pytorch-fi/pytorch-env/bin/activate
python get_inputs.py
deactivate

echo "All files copied successfully."
$LLFI_BUILD_ROOT/bin/instrument --readable -L $ONNX_MLIR_BUILD/Debug/lib -lcruntime -ljson-c -lonnx_proto -lprotobuf model.ll
echo "Instrumented model.ll to model-instrumented.ll"
$LLFI_BUILD_ROOT/bin/profile ./llfi/model-profiling.exe "input${1}_0.pb" "input${1}_1.pb" 0
echo "Generated model-profiling.exe"
$LLFI_BUILD_ROOT/bin/injectfault ./llfi/model-faultinjection.exe "input${1}_0.pb" "input${1}_1.pb"  0
echo "Generated model-faultinjection.exe"

rm -f input*_*.pb

