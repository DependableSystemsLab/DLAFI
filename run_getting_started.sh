source /workspace/chipyard/env.sh
cd /workspace/DLAFI/hw-sim/microbenchmarks
python benchmark_generator.py --chipyard_dir /workspace/chipyard --kernels matmul



SOURCE_FILE="mappings_output.yaml"
# loop over all subdirectories
for d in /workspace/DLAFI/llfi-dlafi/SA_programs/*/ ; do
    if [ -d "$d" ]; then
        echo "Copying to $d"
        cp mappings_output.yaml "$d/SAinput.yaml"
    fi
done

cd /workspace/DLAFI/llfi-dlafi/SA_programs/shufflenet-9
bash compile.sh
bash runllfi.sh 1

cd /workspace/DLAFI/pytorch-fi
source pytorch-env/bin/activate
python main.py --models shufflenet_v2 --num-images 1 --num-iters 5 --sa-dim 16