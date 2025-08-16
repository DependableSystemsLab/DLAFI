from gemmini_benchmark_manager import copy_build_and_run_gemmini_benchmark
from read_waveform import parse_vcd_weight, convert_flat_to_multidim, analyze_waveform


def test_matmul_params(dim_size, SA_dim=16):
    with open("DLAFI_MatMul", "w") as f:
        f.write(f"#define MAT_DIM_K {k}\n")
        f.write(f"#define MAT_DIM_J {j}\n")
        copy_build_and_run_gemmini_benchmark(chipyard_dir, benchmark_name="DLAFI_MatMul", to_build=True, to_run=True)
        values = parse_vcd_weight(chipyard_dir, SA_dim)
        wx, wy = convert_flat_to_multidim(values, SA_dim)
        X, Y, Div_Tiles, d_2unrolls = analyze_waveform(wx, wy, SA_dim)
        if d_2unrolls == None:
            d_2unrolls = (-1, None)  # No 2D unrolling detected
        return X, Y, Div_Tiles, d_2unrolls


def write_conv_params(dim_size, SA_dim=16):
    with open(f"DLAFI_Conv", "w") as f:
        f.write(f"#define KERNEL_DIM {dim_size[0]}\n")
        f.write(f"#define OUT_CHANNELS {dim_size[1]}\n")
        f.write(f"#define IN_CHANNELS {dim_size[2]}\n")
        copy_build_and_run_gemmini_benchmark(chipyard_dir, benchmark_name="DLAFI_Conv", to_build=True, to_run=True)
        values = parse_vcd_weight(chipyard_dir, SA_dim)
        wx, wy = convert_flat_to_multidim(values, SA_dim)
        X, Y, Div_Tiles, d_2unrolls = analyze_waveform(wx, wy, SA_dim)
        if d_2unrolls == None:
            d_2unrolls = (-1, None)  # No 2D unrolling detected
        return X, Y, Div_Tiles, d_2unrolls

def write_dwconv_params(dim_size, SA_dim=16):
    with open(f"DLAFI_DWConv", "w") as f:
        f.write(f"#define KERNEL_DIM {dim_size[0]}\n")
        f.write(f"#define OUT_CHANNELS {dim_size[1]}\n")
        copy_build_and_run_gemmini_benchmark(chipyard_dir, benchmark_name="DLAFI_DWConv", to_build=True, to_run=True)
        values = parse_vcd_weight(chipyard_dir, SA_dim)
        wx, wy = convert_flat_to_multidim(values, SA_dim)
        X, Y, Div_Tiles, d_2unrolls = analyze_waveform(wx, wy, SA_dim)
        if d_2unrolls == None:
            d_2unrolls = (-1, None)  # No 2D unrolling detected
        return X, Y, Div_Tiles, d_2unrolls


def is_similar_mapping(map_large, map_small, target_dim):
    Xl, Yl, _, _ = map_large
    Xs, Ys, _, _ = map_small
    Xl = [x for x in Xl if x != target_dim]
    Yl = [y for y in Yl if y != target_dim]
    return Xl == Xs and Yl == Ys


def find_all_mappings_matmul(Kv, Kc, Vmax, V_min, SA_dim):
    dim_size = [Kv, Kv]
    mapping_base = test_matmul_params(dim_size)
    if mapping_base is None:
        raise RuntimeError("Base mapping failed")
 
    strategies = []
    strategy_base = {
        "strategy_id": "strategy_0",
        "condition": [[-1, "default", -1]],
        "X": mapping_base[0],
        "Y": mapping_base[1],
        "Divisible_tiles": mapping_base[2],
        "dim_with_2d_unroll": mapping_base[3]
    }
    strategies.append(strategy_base)
    strategy_id = 1

    # 2 variable dimensions to check
    for i in range(2):
        dim_size = [Kv, Kv]
        dim_size[i] = Vmax
        mapping_max = test_matmul_params(dim_size)

        dim_size[i] = Vmin
        mapping_min = test_matmul_params(dim_size)

        if not is_similar_mapping(mapping_max, mapping_min, i):
            l = Vmin
            r = Vmax
            boundary = -1
            while l < r:
                mid = (l + r) // 2
                dims = [Kv, Kv]
                dims[i] = mid
                mapping_mid = test_matmul_params(*dims)
                if is_similar_mapping(mapping_mid, mapping_min, i):
                    l = mid + 1
                else:
                    r = mid
            strategy_min = {
                "strategy_id": f"strategy_{strategy_id}",
                "condition": [[i, "leq", l]],
                "X": mapping_min[0],
                "Y": mapping_min[1],
                "Divisible_tiles": mapping_min[2],
                "dim_with_2d_unroll": mapping_min[3]
            }
            strategies.append(strategy_min)
            strategy_id += 1

            strategy_max = {
                "strategy_id": f"strategy_{strategy_id}",
                "condition": [[i, "geq", l + 1]],
                "X": mapping_max[0],
                "Y": mapping_max[1],
                "Divisible_tiles": mapping_max[2],
                "dim_with_2d_unroll": mapping_max[3]
            }
            strategies.append(strategy_max)
            strategy_id += 1
    return {
        "mapping_id": "mapping_matmul",
        "kernel_type": "matmul",
        "num_strategies": len(strategies),
        "strategies": strategies
    }


def find_all_mappings_dwconv(Kv, Kc, Vmax_c, Vmax, V_min, SA_dim):
    dim_size = [Kc, Kv]
    mapping_base = test_conv_params(dim_size)
    if mapping_base is None:
        raise RuntimeError("Base mapping failed")
 
    strategies = []
    strategy_base = {
        "strategy_id": "strategy_0",
        "condition": [[-1, "default", -1]],
        "X": mapping_base[0],
        "Y": mapping_base[1],
        "Divisible_tiles": mapping_base[2],
        "dim_with_2d_unroll": mapping_base[3]
    }
    strategies.append(strategy_base)
    strategy_id = 1

    # check controlled dimension
    for val in range(Vmin, Vmax_c+1):
        dim_size = [Kc, Kv]
        dim_size[0] = val
        mapping_target = test_dwconv_params(dim_size)
        is_similar = True
        if Kc > val:
            is_similar = is_similar_mapping(mapping_base, mapping_target, 0)
        else :
            is_similar = is_similar_mapping(mapping_target, mapping_base, 0)
        if not is_similar:
            strategy_target = {
                "strategy_id": f"strategy_{strategy_id}",
                "condition": [[0, "geq", l + 1]],
                "X": mapping_target[0],
                "Y": mapping_target[1],
                "Divisible_tiles": mapping_target[2],
                "dim_with_2d_unroll": mapping_target[3]
            }
            strategies.append(strategy_max)
            strategy_id += 1

    # 2 variable dimensions to check
    for i in range(1,2):
        dim_size = [Kc, Kv]
        dim_size[i] = Vmax
        mapping_max = test_dwconv_params(dim_size)

        dim_size[i] = Vmin
        mapping_min = test_dwconv_params(dim_size)

        if not is_similar_mapping(mapping_max, mapping_min, i):
            l = Vmin
            r = Vmax
            boundary = -1
            while l < r:
                mid = (l + r) // 2
                dims = [Kv, Kv]
                dims[i] = mid
                mapping_mid = test_dwconv_params(*dims)
                if is_similar_mapping(mapping_mid, mapping_min, i):
                    l = mid + 1
                else:
                    r = mid
            strategy_min = {
                "strategy_id": f"strategy_{strategy_id}",
                "condition": [[i, "leq", l]],
                "X": mapping_min[0],
                "Y": mapping_min[1],
                "Divisible_tiles": mapping_min[2],
                "dim_with_2d_unroll": mapping_min[3]
            }
            strategies.append(strategy_min)
            strategy_id += 1

            strategy_max = {
                "strategy_id": f"strategy_{strategy_id}",
                "condition": [[i, "geq", l + 1]],
                "X": mapping_max[0],
                "Y": mapping_max[1],
                "Divisible_tiles": mapping_max[2],
                "dim_with_2d_unroll": mapping_max[3]
            }
            strategies.append(strategy_max)
            strategy_id += 1
    return {
        "mapping_id": "mapping_dwconv",
        "kernel_type": "dwconv",
        "num_strategies": len(strategies),
        "strategies": strategies
    }

def find_all_mappings_conv(Kv, Kc, Vmax_c, Vmax, V_min, SA_dim):
    dim_size = [Kc, Kv, Kv]
    mapping_base = test_conv_params(dim_size)
    if mapping_base is None:
        raise RuntimeError("Base mapping failed")
 
    strategies = []
    strategy_base = {
        "strategy_id": "strategy_0",
        "condition": [[-1, "default", -1]],
        "X": mapping_base[0],
        "Y": mapping_base[1],
        "Divisible_tiles": mapping_base[2],
        "dim_with_2d_unroll": mapping_base[3]
    }
    strategies.append(strategy_base)
    strategy_id = 1

    for val in range(Vmin, Vmax_c+1):
        dim_size = [Kc, Kv, Kv]
        dim_size[0] = val
        mapping_target = test_conv_params(dim_size)
        is_similar = True
        if Kc > val:
            is_similar = is_similar_mapping(mapping_base, mapping_target, 0)
        else :
            is_similar = is_similar_mapping(mapping_target, mapping_base, 0)
        if not is_similar:
            strategy_target = {
                "strategy_id": f"strategy_{strategy_id}",
                "condition": [[0, "geq", l + 1]],
                "X": mapping_target[0],
                "Y": mapping_target[1],
                "Divisible_tiles": mapping_target[2],
                "dim_with_2d_unroll": mapping_target[3]
            }
            strategies.append(strategy_max)
            strategy_id += 1

    # 2 variable dimensions to check
    for i in range(1,3):
        dim_size = [Kc, Kv, Kv]
        dim_size[i] = Vmax
        mapping_max = test_conv_params(dim_size)

        dim_size[i] = Vmin
        mapping_min = test_conv_params(dim_size)

        if not is_similar_mapping(mapping_max, mapping_min, i):
            l = Vmin
            r = Vmax
            boundary = -1
            while l < r:
                mid = (l + r) // 2
                dims = [Kv, Kv]
                dims[i] = mid
                mapping_mid = test_conv_params(*dims)
                if is_similar_mapping(mapping_mid, mapping_min, i):
                    l = mid + 1
                else:
                    r = mid
            strategy_min = {
                "strategy_id": f"strategy_{strategy_id}",
                "condition": [[i, "leq", l]],
                "X": mapping_min[0],
                "Y": mapping_min[1],
                "Divisible_tiles": mapping_min[2],
                "dim_with_2d_unroll": mapping_min[3]
            }
            strategies.append(strategy_min)
            strategy_id += 1

            strategy_max = {
                "strategy_id": f"strategy_{strategy_id}",
                "condition": [[i, "geq", l + 1]],
                "X": mapping_max[0],
                "Y": mapping_max[1],
                "Divisible_tiles": mapping_max[2],
                "dim_with_2d_unroll": mapping_max[3]
            }
            strategies.append(strategy_max)
            strategy_id += 1
    return {
        "mapping_id": "mapping_conv",
        "kernel_type": "conv",
        "num_strategies": len(strategies),
        "strategies": strategies
    }

def write_mappings_to_yaml(device_config, mappings, filename="output.yaml"):
    data = {
        "deviceOption": device_config,
        "SA_mappings": mappings
    }
    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=False)

if __name__ == "__main__":
        
    # Example usage
    device_config = {
        "SystolicArrayDataflow": "WS",
        "SystolicArrayDimension": 16,
        "deviceType": "SA"
    }
    matmul_mapping = find_all_mappings_matmul(Kv=10, Vmax=128, Vmin=3, SA_dim=16)   
    dwconv_mapping = find_all_mappings_dwconv(Kv=10, Kc=3, Vmax_c=10, Vmax=64, Vmin=3, SA_dim=16)
    conv_mapping = find_all_mappings_conv(Kv=10, Kc=3, Vmax_c=10, Vmax=64, Vmin=3, SA_dim=16)
    write_mappings_to_yaml(device_config, [matmul_mapping, dwconv_mapping, conv_mapping], filename="mappings_output.yaml")