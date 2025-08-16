#!/usr/bin/env python3
import argparse
import yaml
from typing import List, Tuple, Dict, Any

from gemmini_benchmark_manager import copy_build_and_run_gemmini_benchmark
from read_waveform import parse_vcd_weight, convert_flat_to_multidim, analyze_waveform


# ---------- Low-level helper: build+run benchmark, parse waveform ----------
def run_and_parse_waveform(
    header_defines: List[str],
    benchmark_name: str,
    chipyard_dir: str,
    SA_dim: int,
    to_build: bool,
    to_run: bool,
) -> Tuple[List[int], List[int], List[int], Tuple[int, Any]]:
    """
    Writes a small header file with #defines used by the benchmark, builds+runs it,
    parses the VCD waveform, and returns:
      X, Y, Div_Tiles, dim_with_2d_unroll (or (-1, None) if no 2D unroll detected)
    """
    with open(benchmark_name, "w") as f:
        for line in header_defines:
            f.write(f"#define {line}\n")

    copy_build_and_run_gemmini_benchmark(
        chipyard_dir,
        benchmark_name=benchmark_name,
        to_build=to_build,
        to_run=to_run,
    )

    values = parse_vcd_weight(chipyard_dir, SA_dim)
    wx, wy = convert_flat_to_multidim(values, SA_dim)
    X, Y, Div_Tiles, d_2unrolls = analyze_waveform(wx, wy, SA_dim)

    if d_2unrolls is None:
        d_2unrolls = (-1, None)  # Explicit sentinel for “no 2D unroll”

    return X, Y, Div_Tiles, d_2unrolls


# ---------- Concrete kernels (matmul / conv / dwconv) ----------
def test_matmul_params(dim_size: List[int], args) -> Tuple[List[int], List[int], List[int], Tuple[int, Any]]:
    """
    dim_size: [K, J]
    """
    assert len(dim_size) == 2
    K, J = dim_size
    header = [f"MAT_DIM_K {K}", f"MAT_DIM_J {J}"]
    return run_and_parse_waveform(header, "DLAFI_MatMul", args.chipyard_dir, args.SA_dim, args.to_build, args.to_run)


def test_conv_params(dim_size: List[int], args) -> Tuple[List[int], List[int], List[int], Tuple[int, Any]]:
    """
    dim_size: [KERNEL_DIM, OUT_CHANNELS, IN_CHANNELS]  == [Kc, Kv, Kv] as you used
    """
    assert len(dim_size) == 3
    Kc, OC, IC = dim_size
    header = [f"KERNEL_DIM {Kc}", f"OUT_CHANNELS {OC}", f"IN_CHANNELS {IC}"]
    return run_and_parse_waveform(header, "DLAFI_Conv", args.chipyard_dir, args.SA_dim, args.to_build, args.to_run)


def test_dwconv_params(dim_size: List[int], args) -> Tuple[List[int], List[int], List[int], Tuple[int, Any]]:
    """
    Depthwise conv:
    dim_size: [KERNEL_DIM, OUT_CHANNELS] == [Kc, Kv]
    """
    assert len(dim_size) == 2
    Kc, OC = dim_size
    header = [f"KERNEL_DIM {Kc}", f"OUT_CHANNELS {OC}"]
    return run_and_parse_waveform(header, "DLAFI_DWConv", args.chipyard_dir, args.SA_dim, args.to_build, args.to_run)


# ---------- Mapping utilities ----------
def is_similar_mapping(map_large, map_small, target_dim: int) -> bool:
    """
    Compare data reuse mappings ignoring the target_dim in X/Y of the *large* mapping.
    Returns True if (X,Y) except that dimension are identical.
    """
    Xl, Yl, _, _ = map_large
    Xs, Ys, _, _ = map_small

    Xl_wo = [x for x in Xl if x != target_dim]
    Yl_wo = [y for y in Yl if y != target_dim]
    return Xl_wo == Xs and Yl_wo == Ys


# ---------- Search routines to generate mapping strategies ----------
def find_all_mappings_matmul(Kv: int, Vmin: int, Vmax: int, args) -> Dict[str, Any]:
    """
    Matmul dims we sweep: [K, J] (both variable-like)
    """
    dim_size = [Kv, Kv]
    mapping_base = test_matmul_params(dim_size, args)

    strategies = []
    sid = 0

    # Default/base strategy
    strategies.append({
        "strategy_id": f"strategy_{sid}",
        "condition": [[-1, "default", -1]],
        "X": mapping_base[0],
        "Y": mapping_base[1],
        "Divisible_tiles": mapping_base[2],
        "dim_with_2d_unroll": mapping_base[3],
    })
    sid += 1

    # Two variable dimensions: index 0 (K), index 1 (J)
    for i in range(2):
        # Max
        dim_max = [Kv, Kv]
        dim_max[i] = Vmax
        mapping_max = test_matmul_params(dim_max, args)

        # Min
        dim_min = [Kv, Kv]
        dim_min[i] = Vmin
        mapping_min = test_matmul_params(dim_min, args)

        if not is_similar_mapping(mapping_max, mapping_min, i):
            # Binary search boundary where mapping changes from min-like to max-like
            l, r = Vmin, Vmax
            while l < r:
                mid = (l + r) // 2
                dims_mid = [Kv, Kv]
                dims_mid[i] = mid
                mapping_mid = test_matmul_params(dims_mid, args)
                if is_similar_mapping(mapping_mid, mapping_min, i):
                    l = mid + 1
                else:
                    r = mid

            # Everything <= l-1 ~ mapping_min
            strategies.append({
                "strategy_id": f"strategy_{sid}",
                "condition": [[i, "leq", l - 1]],
                "X": mapping_min[0],
                "Y": mapping_min[1],
                "Divisible_tiles": mapping_min[2],
                "dim_with_2d_unroll": mapping_min[3],
            })
            sid += 1

            # Everything >= l ~ mapping_max
            strategies.append({
                "strategy_id": f"strategy_{sid}",
                "condition": [[i, "geq", l]],
                "X": mapping_max[0],
                "Y": mapping_max[1],
                "Divisible_tiles": mapping_max[2],
                "dim_with_2d_unroll": mapping_max[3],
            })
            sid += 1

    return {
        "mapping_id": "mapping_matmul",
        "kernel_type": "matmul",
        "num_strategies": len(strategies),
        "strategies": strategies,
    }


def find_all_mappings_dwconv(Kv: int, Kc: int, Vmin_c: int, Vmax_c: int, Vmin: int, Vmax: int, args) -> Dict[str, Any]:
    """
    Depthwise conv dims: [Kc (controlled), Kv (variable)]
    """
    dim_base = [Kc, Kv]
    mapping_base = test_dwconv_params(dim_base, args)

    strategies = []
    sid = 0

    # Default/base strategy
    strategies.append({
        "strategy_id": f"strategy_{sid}",
        "condition": [[-1, "default", -1]],
        "X": mapping_base[0],
        "Y": mapping_base[1],
        "Divisible_tiles": mapping_base[2],
        "dim_with_2d_unroll": mapping_base[3],
    })
    sid += 1

    # Controlled sweep for Kc over [Vmin_c, Vmax_c]
    for val in range(Vmin_c, Vmax_c + 1):
        if val == Kc:
            continue
        dim_target = [val, Kv]
        mapping_target = test_dwconv_params(dim_target, args)
        # Compare in a direction-agnostic way (stable vs. target)
        similar = True
        if Kc > val:
            similar = is_similar_mapping(mapping_base, mapping_target, 0)
        else:
            similar = is_similar_mapping(mapping_target, mapping_base, 0)
        if not similar:
            strategies.append({
                "strategy_id": f"strategy_{sid}",
                "condition": [[0, "eq", val]],
                "X": mapping_target[0],
                "Y": mapping_target[1],
                "Divisible_tiles": mapping_target[2],
                "dim_with_2d_unroll": mapping_target[3],
            })
            sid += 1

    # Variable dimension for Kv: binary search boundary
    i = 1  # index for Kv
    dim_max = [Kc, Vmax]
    mapping_max = test_dwconv_params(dim_max, args)

    dim_min = [Kc, Vmin]
    mapping_min = test_dwconv_params(dim_min, args)

    if not is_similar_mapping(mapping_max, mapping_min, i):
        l, r = Vmin, Vmax
        while l < r:
            mid = (l + r) // 2
            dims_mid = [Kc, mid]
            mapping_mid = test_dwconv_params(dims_mid, args)
            if is_similar_mapping(mapping_mid, mapping_min, i):
                l = mid + 1
            else:
                r = mid

        strategies.append({
            "strategy_id": f"strategy_{sid}",
            "condition": [[i, "leq", l - 1]],
            "X": mapping_min[0],
            "Y": mapping_min[1],
            "Divisible_tiles": mapping_min[2],
            "dim_with_2d_unroll": mapping_min[3],
        })
        sid += 1

        strategies.append({
            "strategy_id": f"strategy_{sid}",
            "condition": [[i, "geq", l]],
            "X": mapping_max[0],
            "Y": mapping_max[1],
            "Divisible_tiles": mapping_max[2],
            "dim_with_2d_unroll": mapping_max[3],
        })
        sid += 1

    return {
        "mapping_id": "mapping_dwconv",
        "kernel_type": "dwconv",
        "num_strategies": len(strategies),
        "strategies": strategies,
    }


def find_all_mappings_conv(Kv: int, Kc: int, Vmin_c: int, Vmax_c: int, Vmin: int, Vmax: int, args) -> Dict[str, Any]:
    """
    Conv dims (following your layout): [Kc (controlled), OC (variable), IC (variable)].
    We’ll use OC=Kv and IC=Kv for the sweeps (symmetry).
    """
    dim_base = [Kc, Kv, Kv]
    mapping_base = test_conv_params(dim_base, args)

    strategies = []
    sid = 0

    # Default/base strategy
    strategies.append({
        "strategy_id": f"strategy_{sid}",
        "condition": [[-1, "default", -1]],
        "X": mapping_base[0],
        "Y": mapping_base[1],
        "Divisible_tiles": mapping_base[2],
        "dim_with_2d_unroll": mapping_base[3],
    })
    sid += 1

    # Controlled sweep for Kc over [Vmin_c, Vmax_c]
    for val in range(Vmin_c, Vmax_c + 1):
        if val == Kc:
            continue
        dim_target = [val, Kv, Kv]
        mapping_target = test_conv_params(dim_target, args)
        similar = True
        if Kc > val:
            similar = is_similar_mapping(mapping_base, mapping_target, 0)
        else:
            similar = is_similar_mapping(mapping_target, mapping_base, 0)
        if not similar:
            strategies.append({
                "strategy_id": f"strategy_{sid}",
                "condition": [[0, "eq", val]],
                "X": mapping_target[0],
                "Y": mapping_target[1],
                "Divisible_tiles": mapping_target[2],
                "dim_with_2d_unroll": mapping_target[3],
            })
            sid += 1

    # Two variable dims: OC (index 1), IC (index 2)
    for i in (1, 2):
        dim_max = [Kc, Kv, Kv]
        dim_max[i] = Vmax
        mapping_max = test_conv_params(dim_max, args)

        dim_min = [Kc, Kv, Kv]
        dim_min[i] = Vmin
        mapping_min = test_conv_params(dim_min, args)

        if not is_similar_mapping(mapping_max, mapping_min, i):
            l, r = Vmin, Vmax
            while l < r:
                mid = (l + r) // 2
                dims_mid = [Kc, Kv, Kv]
                dims_mid[i] = mid
                mapping_mid = test_conv_params(dims_mid, args)
                if is_similar_mapping(mapping_mid, mapping_min, i):
                    l = mid + 1
                else:
                    r = mid

            strategies.append({
                "strategy_id": f"strategy_{sid}",
                "condition": [[i, "leq", l - 1]],
                "X": mapping_min[0],
                "Y": mapping_min[1],
                "Divisible_tiles": mapping_min[2],
                "dim_with_2d_unroll": mapping_min[3],
            })
            sid += 1

            strategies.append({
                "strategy_id": f"strategy_{sid}",
                "condition": [[i, "geq", l]],
                "X": mapping_max[0],
                "Y": mapping_max[1],
                "Divisible_tiles": mapping_max[2],
                "dim_with_2d_unroll": mapping_max[3],
            })
            sid += 1

    return {
        "mapping_id": "mapping_conv",
        "kernel_type": "conv",
        "num_strategies": len(strategies),
        "strategies": strategies,
    }


# ---------- Top-level generator ----------
def generate_mappings(args) -> Dict[str, Any]:
    """
    Produces the final YAML-able dict with device config + all kernel mappings.
    Safe to import and call from another script.
    """
    device_config = {
        "SystolicArrayDataflow": "WS",
        "SystolicArrayDimension": args.SA_dim,
        "deviceType": "SA",
    }

    matmul_mapping = find_all_mappings_matmul(Kv=args.Kv, Vmin=args.Vmin, Vmax=args.Vmax, args=args)
    dwconv_mapping = find_all_mappings_dwconv(Kv=args.Kv, Kc=args.Kc, Vmin_c=args.Vmin_c, Vmax_c=args.Vmax_c,
                                              Vmin=args.Vmin, Vmax=args.Vmax, args=args)
    conv_mapping = find_all_mappings_conv(Kv=args.Kv, Kc=args.Kc, Vmin_c=args.Vmin_c, Vmax_c=args.Vmax_c,
                                          Vmin=args.Vmin, Vmax=args.Vmax, args=args)

    return {
        "deviceOption": device_config,
        "SA_mappings": [matmul_mapping, dwconv_mapping, conv_mapping],
    }


def write_mappings_to_yaml(payload: Dict[str, Any], filename: str) -> None:
    with open(filename, "w") as f:
        yaml.dump(payload, f, sort_keys=False)


# ---------- CLI ----------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate YAML mapping configurations for SA kernels")

    # Problem sizes / sweep ranges
    p.add_argument("--Kv", type=int, default=10, required=True,
                   help="Base variable dim value (e.g., out/in channels for conv; K or J for matmul)")
    p.add_argument("--Kc", type=int, default=5, required=True,
                   help="Base kernel dim (e.g., KERNEL_DIM)")
    p.add_argument("--Vmin", type=int, default=3, required=True,
                   help="Minimum value for variable dims")
    p.add_argument("--Vmax", type=int, default=40, required=True,
                   help="Maximum value for variable dims")
    p.add_argument("--Vmin_c", type=int, default=3, required=True,
                   help="Minimum value for controlled (kernel) dim")
    p.add_argument("--Vmax_c", type=int, default=10, required=True,
                   help="Maximum value for controlled (kernel) dim")

    # Build/run/system
    p.add_argument("--chipyard_dir", type=str, default="/workspace/chipyard", required=True,
                   help="Path to Chipyard root used by benchmark manager")
    p.add_argument("--SA_dim", type=int, default=16, help="Systolic array dimension")
    # Default these to True as requested; to disable, set explicitly with e.g. --to_build False
    p.add_argument("--to_build", type=lambda x: str(x).lower() not in ("0","false","no","off"), default=True,
                   help="Trigger build (default: True). Pass 'False' to disable.")
    p.add_argument("--to_run", type=lambda x: str(x).lower() not in ("0","false","no","off"), default=True,
                   help="Trigger run (default: True). Pass 'False' to disable.")

    # Output
    p.add_argument("--output", type=str, default="mappings_output.yaml", help="YAML output path")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    payload = generate_mappings(args)
    write_mappings_to_yaml(payload, args.output)
    print(f"Wrote mappings to {args.output}")


if __name__ == "__main__":
    main()
