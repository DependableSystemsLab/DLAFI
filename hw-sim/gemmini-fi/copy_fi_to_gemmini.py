import os
import shutil
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python copy_to_gemmini.py <chipyard_dir>")
        sys.exit(1)

    chipyard_dir = sys.argv[1]

    # fie: destination directory
    files_to_copy = {
        "fi_cmd_args.h" : "generators/gemmini/software/gemmini-rocc-tests/include",
        "gemmini.h" : "generators/gemmini/software/gemmini-rocc-tests/include",
        "Arithmetic.scala" : "generators/gemmini/src/main/scala/gemmini",
        "Configs.scala" : "generators/gemmini/src/main/scala/gemmini",
        "Controller.scala" : "generators/gemmini/src/main/scala/gemmini",
        "ExecuteController.scala" : "generators/gemmini/src/main/scala/gemmini", 
        "GemminiISA.scala" : "generators/gemmini/src/main/scala/gemmini",
        "Mesh.scala" : "generators/gemmini/src/main/scala/gemmini",
        "MeshWithDelays.scala" : "generators/gemmini/src/main/scala/gemmini", 
        "PE.scala" : "generators/gemmini/src/main/scala/gemmini",
        "Tile.scala" : "generators/gemmini/src/main/scala/gemmini",
        "build-verilator.sh" : "generators/gemmini/scripts/build-verilator.sh",
        "variables.mk" : ""
    }


    for file, rel_path in files_to_copy.items():
        src = file
        dst = os.path.join(chipyard_dir, rel_path)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"Copied {src} to {dst}")
        else:
            print(f"File not found: {src}")

if __name__ == "__main__":
    main()