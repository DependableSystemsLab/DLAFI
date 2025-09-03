rm -rf llfi*
mkdir FIutil

source_dir="../imagenet_samples"
destination_dir="."

# LLFI_BUILD_ROOT="/home/msadati/LLFI"
end=$(( $1 - 1 ))
# Loop through the files you want to copy
# Adjust the range {1..10} as needed
for i in $(seq 0 $end); do
    source_file="input_$((i + 0)).pb"  # Construct the source file name
    destination_file="input_$i.pb"      # Construct the destination file name
    file_name="input_${i}.pb" # Construct the file name
    # Check if the file exists before attempting to copy
    if [ -f "$source_dir/$source_file" ]; then
        cp "$source_dir/$source_file" "$destination_dir/$destination_file"
        echo "Copied $source_file to $destination_dir/$destination_file"
    else
        echo "$source_file does not exist, skipping."
    fi
done
echo "All files copied successfully."
$LLFI_BUILD_ROOT/bin/instrument --readable -L $ONNX_MLIR_BUILD/Debug/lib -lcruntime -ljson-c -lonnx_proto -lprotobuf model.ll
echo "Instrumented model.ll to model-instrumented.ll"
$LLFI_BUILD_ROOT/bin/profile ./llfi/model-profiling.exe  $1   0 profiling
echo "Generated model-profiling.exe"
$LLFI_BUILD_ROOT/bin/injectfault ./llfi/model-faultinjection.exe  $1  0 injectfault
echo "Generated model-faultinjection.exe"

for i in $(seq 0 $end); do
    file_name="input_${i}.pb"
    rm "$destination_dir/$file_name"
    echo "removed $file_name to $destination_dir"
done

#  input_0.pb  input_1.pb  input_2.pb  input_3.pb  input_4.pb  input_5.pb  input_6.pb  input_7.pb  input_8.pb  input_9.pb  input_10.pb  input_11.pb  input_12.pb  input_13.pb  input_14.pb  input_15.pb  input_16.pb  input_17.pb  input_18.pb  input_19.pb  input_20.pb  input_21.pb  input_22.pb  input_23.pb  input_24.pb  input_25.pb  input_26.pb  input_27.pb  input_28.pb  input_29.pb  input_30.pb  input_31.pb  input_32.pb  input_33.pb  input_34.pb  input_35.pb  input_36.pb  input_37.pb  input_38.pb  input_39.pb  input_40.pb  input_41.pb  input_42.pb  input_43.pb  input_44.pb  input_45.pb  input_46.pb  input_47.pb  input_48.pb  input_49.pb  input_50.pb  input_51.pb  input_52.pb  input_53.pb  input_54.pb  input_55.pb  input_56.pb  input_57.pb  input_58.pb  input_59.pb  input_60.pb  input_61.pb  input_62.pb  input_63.pb  input_64.pb  input_65.pb  input_66.pb  input_67.pb  input_68.pb  input_69.pb  input_70.pb  input_71.pb  input_72.pb  input_73.pb  input_74.pb  input_75.pb  input_76.pb  input_77.pb  input_78.pb  input_79.pb  input_80.pb  input_81.pb  input_82.pb  input_83.pb  input_84.pb  input_85.pb  input_86.pb  input_87.pb  input_88.pb  input_89.pb  input_90.pb  input_91.pb  input_92.pb  input_93.pb  input_94.pb  input_95.pb  input_96.pb  input_97.pb  input_98.pb  input_99.pb  input_100.pb
