#!/bin/bash
set -e
set -o pipefail

echo "========= DLAFI HARDWARE-AWARE BENCHMARKING STARTED ========="

# ========= SOURCE CHIPYARD ENVIRONMENT =========
if [ -f "$CHIPYARD_ROOT/env.sh" ]; then
    echo "Sourcing Chipyard environment..."
    source "$CHIPYARD_ROOT/env.sh"
else
    echo "Error: $CHIPYARD_ROOT/env.sh not found!"
    exit 1
fi

# ========= GENERATE BENCHMARK MAPPINGS =========
echo "Generating mapping abstractions..."
cd "$DLAFI_ROOT/hw-sim/microbenchmarks"

python3 benchmark_generator.py \
    --chipyard_dir "$CHIPYARD_ROOT" \
    --kernels matmul

# ========= DISTRIBUTE GENERATED MAPPINGS =========
SOURCE_FILE="mappings_output.yaml"
LLFI_SA_DIR="$DLAFI_ROOT/llfi-dlafi/SA_programs"

if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: $SOURCE_FILE not found!"
    exit 1
fi

echo "Distributing mappings to all SA_programs subdirectories..."
for d in "$LLFI_SA_DIR"/*/; do
    if [ -d "$d" ]; then
        echo "→ Copying to: $d"
        cp "$SOURCE_FILE" "$d/SAinput.yaml"
    fi
done

# ========= COMPILE AND RUN LLFI BENCHMARK (shufflenet-v2) =========
TARGET_DIR="$LLFI_SA_DIR/shufflenet-v2-10"
echo "Running LLFI benchmark in: $TARGET_DIR"
cd "$TARGET_DIR"

if [ -x "./compile.sh" ]; then
    bash compile.sh
else
    echo "Warning: compile.sh not found or not executable."
fi

if [ -x "./runllfi.sh" ]; then
    bash runllfi.sh 1
else
    echo "Warning: runllfi.sh not found or not executable."
fi

# ========= RUN PYTORCH-FI INFERENCE (shufflenet-v2) =========
echo "Running PyTorch-FI evaluation..."
cd "$DLAFI_ROOT/pytorch-fi"

if [ ! -d "pytorch-env" ]; then
    echo "Error: pytorch-env virtual environment not found!"
    exit 1
fi

source pytorch-env/bin/activate

python3 main.py \
    --models shufflenet_v2 \
    --num-images 1 \
    --num-iters 5 \
    --sa-dim 16

echo "========= DLAFI HARDWARE-AWARE BENCHMARKING COMPLETE ========="
