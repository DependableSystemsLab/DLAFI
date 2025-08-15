import os
import shutil
import sys
import subprocess

def copy_build_and_run_gemmini_benchmark(chipyard_dir, benchmark_name=None, to_build=True, to_run=True):
    """
    Copy DLAFI benchmarks and Makefile to the Gemmini bareMetalC directory,
    then source env.sh and run build.sh, and optionally run a benchmark.
    """
    files_to_copy = {
        "DLAFI_Conv.c": "generators/gemmini/software/gemmini-rocc-tests/bareMetalC",
        "DLAFI_Conv.h": "generators/gemmini/software/gemmini-rocc-tests/bareMetalC",
        "DLAFI_DWConv.c": "generators/gemmini/software/gemmini-rocc-tests/bareMetalC",
        "DLAFI_DWConv.h": "generators/gemmini/software/gemmini-rocc-tests/bareMetalC",
        "DLAFI_MatMul.c": "generators/gemmini/software/gemmini-rocc-tests/bareMetalC",
        "DLAFI_MatMul.h": "generators/gemmini/software/gemmini-rocc-tests/bareMetalC",
        "Makefile": "generators/gemmini/software/gemmini-rocc-tests/bareMetalC"
    }

    for filename, rel_path in files_to_copy.items():
        src = filename
        dst = os.path.join(chipyard_dir, rel_path)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"File not found: {src}")

    if not to_build:
        print("Skipping build step.")
        return

    # Build step
    gemmini_sw_dir = os.path.join(chipyard_dir, "generators/gemmini/software/gemmini-rocc-tests")
    env_sh_path = os.path.join(chipyard_dir, "env.sh")
    build_cmd = f"source {env_sh_path} && ./build.sh"
    print(f"Running: source {env_sh_path} && ./build.sh in {gemmini_sw_dir}")
    subprocess.run(["bash", "-c", build_cmd], cwd=gemmini_sw_dir, check=True)

    if not to_run:
        print("Skipping run step.")
        return

    # Run step
    gemmini_dir = os.path.join(chipyard_dir, "generators/gemmini")
    run_cmd = f"source {env_sh_path} && time ./scripts/run-verilator.sh {benchmark_name} --debug"
    print(f"Running: {run_cmd} in {gemmini_dir}")
    subprocess.run(["bash", "-c", run_cmd], cwd=gemmini_dir, check=True)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python copy_benchmarks_to_gemmini.py <chipyard_dir> <benchmark_name>")
        sys.exit(1)
    chipyard_dir = sys.argv[1]
    benchmark_name = sys.argv[2]
    copy_build_and_run_gemmini_benchmark(chipyard_dir, benchmark_name, to_build=True, to_run=True)