#!/bin/bash
set -e
set -o pipefail

echo "========= CHIPYARD + GEMMINI INSTALLATION STARTED ========="

# ========= PREPARE SYSTEM =========
echo "[1/7] Updating package sources and installing dependencies..."
cd /workspace
sed -i 's|http://security.ubuntu.com/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list
apt-get update -qq
apt-get install -y libpixman-1-dev

# ========= CLONE CHIPYARD =========
echo "[2/7] Cloning Chipyard repository..."
if [ -d "$CHIPYARD_ROOT" ]; then
    echo "Chipyard directory already exists at $CHIPYARD_ROOT — skipping clone."
else
    git clone https://github.com/ucb-bar/chipyard.git "$CHIPYARD_ROOT"
fi

cd "$CHIPYARD_ROOT"
echo "Checking out stable Chipyard commit..."
git fetch --all
git checkout 117624d8eea27bafd613eec09e9b9b3e31239e08

echo "Initializing Chipyard submodules (no riscv-tools)..."
./scripts/init-submodules-no-riscv-tools.sh

# ========= BUILD TOOLCHAINS =========
echo "[3/7] Building esp-tools toolchain "
./scripts/build-toolchains.sh esp-tools

echo "Loading Chipyard environment..."
source env.sh

# ========= GEMMINI SETUP =========
echo "[4/7] Checking out Gemmini v0.6.3..."
cd generators/gemmini
git fetch --all
git checkout v0.6.3
git submodule update --init --recursive

# ========= PATCH SPIKE (RISC-V ISA SIMULATOR) =========
echo "[5/7] Patching Spike (riscv-isa-sim) for Gemmini..."
cd "$CHIPYARD_ROOT/toolchains/esp-tools/riscv-isa-sim/build"
git fetch --all
git checkout 090e82c473fd28b4eb2011ffcd771ead6076faab
make -j"$(nproc)"
make install

# ========= FINAL GEMMINI BUILD =========
echo "[6/7] Running Gemmini setup scripts..."
cd "$CHIPYARD_ROOT/generators/gemmini"
./scripts/setup-paths.sh

# ========= GEMMINI TESTS =========
echo "[7/7] Building and simulating Gemmini tests..."
cd "$CHIPYARD_ROOT/generators/gemmini/software/gemmini-rocc-tests"
./build.sh
./scripts/build-verilator.sh

echo "========= CHIPYARD + GEMMINI INSTALLATION COMPLETE ========="

# -------------------------------------------------------------------------
# ========= DLAFI INTEGRATION: COPY FAULT INJECTION FILES AND REBUILD ======
# -------------------------------------------------------------------------
echo "========= DLAFI-GEMMINI INTEGRATION STARTED ========="

if [ -z "$DLAFI_ROOT" ]; then
    echo "Error: DLAFI_ROOT is not set. Please export it before running."
    exit 1
fi

cd "$DLAFI_ROOT/hw-sim/gemmini-fi"
if [ ! -f "copy_fi_to_gemmini.py" ]; then
    echo "Error: copy_fi_to_gemmini.py not found in $DLAFI_ROOT/hw-sim/gemmini-fi"
    exit 1
fi

echo "Copying DLAFI fault injection hooks into Gemmini..."
python3 copy_fi_to_gemmini.py "$CHIPYARD_ROOT"

echo "Rebuilding Gemmini in debug mode for fault injection (will take several hours!)... "
cd "$CHIPYARD_ROOT"
source env.sh
cd generators/gemmini
./scripts/build-verilator.sh --debug

echo "========= DLAFI-GEMMINI INTEGRATION COMPLETE ========="
