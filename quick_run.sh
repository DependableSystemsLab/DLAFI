#!/bin/bash
set -e
set -o pipefail

# ========= USAGE =========
usage() {
  echo "Usage: $0 [options]"
  echo
  echo "Options (can combine multiple phases):"
  echo "  --hw-sim       Run hardware simulation phase (Chipyard + mapping generation)"
  echo "  --llvm-fi      Run LLVM-level fault injection (LLFI)"
  echo "  --pytorch-fi   Run PyTorch-level inference fault injection"
  echo "  --all          Run all phases (default)"
  echo
  echo "Examples:"
  echo "  $0 --hw-sim"
  echo "  $0 --llvm-fi --pytorch-fi"
  echo "  $0 --all"
  exit 1
}

if [ $# -eq 0 ]; then
  usage
fi

# ========= PARSE OPTIONS =========
RUN_HW_SIM=false
RUN_LLFI=false
RUN_PYTORCH=false

for arg in "$@"; do
  case $arg in
    --hw-sim) RUN_HW_SIM=true ;;
    --llvm-fi) RUN_LLFI=true ;;
    --pytorch-fi) RUN_PYTORCH=true ;;
    --all)
      RUN_HW_SIM=true
      RUN_LLFI=true
      RUN_PYTORCH=true
      ;;
    -h|--help) usage ;;
    *)
      echo "Unknown option: $arg"
      usage
      ;;
  esac
done

echo "========= DLAFI PIPELINE STARTED ========="

# ========= SOURCE CHIPYARD ENVIRONMENT =========
if [ -f "$CHIPYARD_ROOT/env.sh" ]; then
  echo "Sourcing Chipyard environment..."
  source "$CHIPYARD_ROOT/env.sh"
else
  echo "Error: $CHIPYARD_ROOT/env.sh not found!"
  exit 1
fi

# ---------------------------------------------------------------------------
# PHASE 1: HARDWARE SIMULATION / MAPPING GENERATION
# ---------------------------------------------------------------------------
if [ "$RUN_HW_SIM" = true ]; then
  echo "========= PHASE 1: HARDWARE SIMULATION ========="
  cd "$DLAFI_ROOT/hw-sim/microbenchmarks"
  python3 benchmark_generator.py \
    --chipyard_dir "$CHIPYARD_ROOT" \
    --kernels matmul

  SOURCE_FILE="mappings_output.yaml"
  LLFI_SA_DIR="$DLAFI_ROOT/llfi-dlafi/SA_programs"

  if [ ! -f "$SOURCE_FILE" ]; then
      echo "Error: $SOURCE_FILE not found!"
      exit 1
  fi

  echo "Distributing mappings to all SA_programs subdirectories..."
  for d in "$LLFI_SA_DIR"/*/; do
      if [ -d "$d" ]; then
          echo " Copying to: $d"
          cp "$SOURCE_FILE" "$d/SAinput.yaml"
      fi
  done
fi

# ---------------------------------------------------------------------------
# PHASE 2: LLVM-LEVEL FAULT INJECTION (LLFI)
# ---------------------------------------------------------------------------
if [ "$RUN_LLFI" = true ]; then
  echo "========= PHASE 2: LLVM-LEVEL FAULT INJECTION ========="
  TARGET_DIR="$DLAFI_ROOT/llfi-dlafi/SA_programs/shufflenet-v2-10"
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
fi

# ---------------------------------------------------------------------------
# PHASE 3: PYTORCH-LEVEL INFERENCE FAULT INJECTION
# ---------------------------------------------------------------------------
if [ "$RUN_PYTORCH" = true ]; then
  echo "========= PHASE 3: PYTORCH-LEVEL INFERENCE ========="
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
fi

echo "========= DLAFI PIPELINE COMPLETE ========="
