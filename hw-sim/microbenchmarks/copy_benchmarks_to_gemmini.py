import os
import shutil
import sys
import subprocess

def copy_and_build_gemmini_benchmarks(chipyard_dir, build=True):
    """
    Copy DLAFI benchmarks and Makefile to the Gemmini bareMetalC directory,
    then source env.sh and run build.sh.
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

    if not build:
        print("Skipping build step.")
        return

    # Source env.sh and run build.sh in the correct directory
    gemmini_sw_dir = os.path.join(chipyard_dir, "generators/gemmini/software/gemmini-rocc-tests")
    env_sh_path = os.path.join(chipyard_dir, "env.sh")
    build_cmd = f"source {env_sh_path} && ./build.sh"
    print(f"Running: source {env_sh_path} && ./build.sh in {gemmini_sw_dir}")
    subprocess.run(["bash", "-c", build_cmd], cwd=gemmini_sw_dir, check=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python copy_benchmarks_to_gemmini.py <chipyard_dir>")
        sys.exit(1)
    chipyard_dir = sys.argv[1]
    copy_and_build_gemmini_benchmarks(chipyard_dir, True)