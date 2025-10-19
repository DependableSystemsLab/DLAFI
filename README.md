# DLAFI

**Software-Based Fault Injection for Permanent Faults in Deep Learning Accelerators**

This repository contains DLAFI, a hardware-aware, software-level fault-injection (FI) framework that models permanent faults in systolic arrays (SAs) with the speed of software-level injection and the accuracy of hardware simulation. The accompanying [paper](https://blogs.ubc.ca/dependablesystemslab/2025/07/19/dlafi-software-based-fault-injection-for-permanent-faults-in-deep-learning-accelerators/) (Accepted in ISSRE 2025) describes the approach, experiments, and results in detail.

---

## Table of Contents
- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [How to install](#how-to-install)
  - [Build inside Docker](#build-inside-docker)
  - [Environment variables](#environment-variables)
- [Getting started (quick run)](#getting-started-quick-run)
   - [Basic usage](#basic-usage)
   - [What the script does internally](#what-the-script-does-internally)

---

## Overview
DLAFI extracts microarchitectural mapping abstractions of a systolic array (SA) via a small set of hardware microbenchmarks run in RTL simulation (Gemmini). These mapping abstractions are then used to perform hardware-aware fault injection at the LLVM IR level across large ML models, combining accuracy and scale.

If you want to understand the theoretical details, experimental setup, and results, see the [paper](https://blogs.ubc.ca/dependablesystemslab/2025/07/19/dlafi-software-based-fault-injection-for-permanent-faults-in-deep-learning-accelerators/).

---

## Repository layout
```
/ (repo root)
├─ hw-sim/                # microbenchmarks, generator scripts
├─ llfi-dlafi/            # LLFI integration and SA_programs (SA microbenchmarks)
├─ pytorch-fi/            # application-level PyTorch FI and evaluation scripts
├─ install_script.sh      # top-level installer (builds LLVM, ONNX-MLIR, DLAFI)
├─ install_hw_sim_script.sh  # installs Chipyard + Gemmini (HW sim support)
├─ quick_run.sh    # installation scripts and getting strated script
├─ Dockerfile
└─ README.md (this file)
```

---

## Requirements
- Docker (recommended and tested) OR Ubuntu 20.04/22.04 host with `sudo` privileges
- At least **100 GB** disk space (more recommended for toolchains, build artifacts)
- Build-time RAM and CPU: building LLVM/ONNX-MLIR benefits from multiple cores; use `-j$(nproc)` where applicable.

Recommended: run inside the repository's Docker image which already contains most system packages used in the paper's artifact.

---

## How to install
**Docker build**: The repo contains `Dockerfile` and `install_script.sh` to help automation.

### Building the Docker
1. Build the Docker image from the top-level `Dockerfile`:
```bash
docker build -t dlafi_image .
```
2. Run a container and mount the repo for persistence:
```bash
docker run -it --name dlafi_container dlafi_image
```


### Environment variables
Set these paths into your docker's `~/.bashrc` before running install scripts. They are used throughout the scripts and make the workflow portable.

```bash
# LLVM (source/build)
export LLVM_SRC=/workspace/llvm-project
export LLVM_DST_ROOT=$LLVM_SRC/build
export MLIR_DIR=$LLVM_DST_ROOT/lib/cmake/mlir

# ONNX-MLIR
export ONNX_MLIR_SRC=/workspace/onnx-mlir
export ONNX_MLIR_BUILD=$ONNX_MLIR_SRC/build

# LLFI/DLAFI
export LLFI_BUILD_ROOT=/workspace/llfi-build
export DLAFI_ROOT=/workspace/DLAFI

# Chipyard (hardware sims)
export CHIPYARD_ROOT=/workspace/chipyard
```
You can save these lines to `~/.bashrc` and source it:
```bash
source ~/.bashrc
```

3. Inside the container, run the installer for DLAFI and its HW simulation compoenent :
```bash
git clone https://github.com/ManiSadati/DLAFI.git $DLAFI_ROOT

cd $DLAFI_ROOT

# installing LLVM, ONNX MLIR, PyTorch environment, and DLAFI (might take 1-2 hours)
bash install_script.sh

# (Optional) To install Chipyard + Gemmini for HW sims, Note that it will take several hours to install:
bash install_hw_sim_script.sh
```

---
## Getting started (quick run)
After completing installation, you can execute a minimal DLAFI flow — consisting of **mapping generation → LLVM‑level fault injection → PyTorch‑FI evaluation** — using the helper script `quick_run.sh`.

### Basic usage
To run only the LLVM‑level FI and compare it with PyTorch‑FI results:
```bash
bash quick_run.sh --llvm-fi --pytorch-fi
```

If you have also installed the hardware simulation components (Chipyard + Gemmini), you can execute the full end‑to‑end flow:
```bash
bash quick_run.sh --all
```

### What the script does internally
The script leverages the environment variables defined earlier to coordinate the following phases:

1. **Hardware‑aware mapping generation**  
   Generates systolic array (SA) mapping abstractions using hardware microbenchmarks.  
   You can also run this manually:
   ```bash
   cd "$DLAFI_ROOT/hw-sim/microbenchmarks"
   python3 benchmark_generator.py \
       --chipyard_dir "$CHIPYARD_ROOT" \
       --kernels matmul
   ```
   The `--kernels` flag can be changed to other supported kernels for additional benchmarks.

2. **Mapping distribution**  
   The generated file `mappings_output.yaml` is automatically copied into each benchmark directory under:
   ```
   $DLAFI_ROOT/llfi-dlafi/SA_programs/*/SAinput.yaml
   ```

3. **LLVM‑level fault injection (LLFI)**  
   Compiles and runs fault injection for each benchmark. You can also do this manually for a specific model:
   ```bash
   cd "$DLAFI_ROOT/llfi-dlafi/SA_programs/shufflenet-v2-10" # Try other folders for experimenting with different benchmarks
   bash compile.sh
   bash runllfi.sh 1   # Increase the argument to test multiple inputs
   ```

4. **PyTorch‑FI evaluation**  
   Executes the corresponding PyTorch‑level fault injection and reports results:
   ```bash
   cd "$DLAFI_ROOT/pytorch-fi"
   source pytorch-env/bin/activate
   python3 main.py \
       --models shufflenet_v2 \   # You can experiment with other models and even load your own pretrained model
       --num-images 1 \
       --num-iters 5 \
       --sa-dim 16
   ```

These steps collectively reproduce the hardware‑aware and software‑level FI workflow used in the paper.



---

*Last updated: [October 18th, 2025]*

