#!/usr/bin/env python3
import argparse
import yaml
import os
from typing import List, Tuple, Dict, Any

from gemmini_benchmark_manager import copy_build_and_run_gemmini_benchmark
from read_waveform import parse_vcd_weight, convert_flat_to_multidim, analyze_waveform
from typing import Optional

# ---------- Low-level helper: build+run benchmark, parse waveform ----------
def run_and_parse_waveform(
    dim_size: List[int],
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
    print("\n","=="*40,"\n")
    with open(f"{benchmark_name}.h", "w") as f:
        for line in header_defines:
            f.write(f"#define {line}\n")
    src_c = f"{benchmark_name}.c"
    try:
        os.utime(src_c, None)  # touch
    except FileNotFoundError:
        pass
    print(f"Generated header for {benchmark_name} with defines: {header_defines}")
    copy_build_and_run_gemmini_benchmark(
        chipyard_dir,
        benchmark_name=benchmark_name,
        to_build=to_build,
        to_run=to_run,
    )
    values = parse_vcd_weight(chipyard_dir, SA_dim)
    wx, wy, indices = convert_flat_to_multidim(dim_size, values, SA_dim)
    X, Y, Div_Tiles, d_2unrolls = analyze_waveform(wx, wy, indices, dim_size, SA_dim)
    if d_2unrolls is None:
        d_2unrolls = (-1, None)  # Explicit sentinel for “no 2D unroll”
    print(f"Extracted mapping: X={X}, Y={Y}, Div_Tiles={Div_Tiles}, 2D_unroll={d_2unrolls}")
    return X, Y, Div_Tiles, d_2unrolls


# ---------- Concrete kernels (matmul / conv / dwconv) ----------
def test_matmul_params(dim_size: List[int], args) -> Tuple[List[int], List[int], List[int], Tuple[int, Any]]:
    """
    dim_size: [J, K]
    """
    assert len(dim_size) == 2
    K, J = dim_size
    header = [f"MAT_DIM_J {K}", f"MAT_DIM_K {J}"]
    return run_and_parse_waveform(dim_size, header, "DLAFI_MatMul", args.chipyard_dir, args.SA_dim, args.to_build, args.to_run)


def test_conv_params(dim_size: List[int], args) -> Tuple[List[int], List[int], List[int], Tuple[int, Any]]:
    """
    dim_size: [KERNEL_DIM, IN_CHANNELS, OUT_CHANNELS]  == [Kc, Kv, Kv] as you used
    """
    assert len(dim_size) == 3
    Kc, OC, IC = dim_size
    header = [f"KERNEL_DIM {Kc}", f"IN_CHANNELS {OC}", f"OUT_CHANNELS {IC}"]
    modified_dim_size = [dim_size[0], dim_size[0], dim_size[1], dim_size[2]]  # Use only Kc and Kv for the benchmark
    return run_and_parse_waveform(modified_dim_size, header, "DLAFI_Conv", args.chipyard_dir, args.SA_dim, args.to_build, args.to_run)


def test_dwconv_params(dim_size: List[int], args) -> Tuple[List[int], List[int], List[int], Tuple[int, Any]]:
    """
    Depthwise conv:
    dim_size: [KERNEL_DIM, OUT_CHANNELS] == [Kc, Kv]
    """
    assert len(dim_size) == 2
    Kc, OC = dim_size
    header = [f"KERNEL_DIM {Kc}", f"OUT_CHANNELS {OC}"]
    return run_and_parse_waveform(dim_size, header, "DLAFI_DWConv", args.chipyard_dir, args.SA_dim, args.to_build, args.to_run)


# ---------- Mapping utilities ----------
def is_similar_mapping(map_large, map_small, target_dim: int) -> bool:
    """
    Compare data reuse mappings.
    Returns True if (X,Y) except that dimension are identical.
    """
    Xl, Yl, _, _ = map_large
    Xs, Ys, _, _ = map_small

    return Xl == Xs and Yl == Ys


# ---------- Search routines to generate mapping strategies ----------
def find_all_mappings_matmul(Kv: int, Vmin: int, Vmax: int, args) -> Dict[str, Any]:
    """
    Matmul dims we sweep: [J, K] (both variable-like)
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
    return {
        "mapping_id": "mapping_matmul",
        "kernel_type": "matmul",
        "num_strategies": len(strategies),
        "strategies": strategies,
    }

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
            "condition": [[i + 1, "leq", l - 1]], # +1 Index shift for OC
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

    # Two variable dims: IC (index 1), OC (index 2)
    for i in range(1, 3):
        dim_min = [Kc, Kv, Kv]
        dim_min[i] = Vmin
        mapping_min = test_conv_params(dim_min, args)

        dim_max = [Kc, Kv, Kv]
        dim_max[i] = Vmax
        mapping_max = test_conv_params(dim_max, args)
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
                "condition": [[i + 1, "leq", l - 1]], # +1 Index shift for IC/OC
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



# ---------- CLI ----------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate YAML mapping configurations for SA kernels")

    # Problem sizes / sweep ranges
    p.add_argument("--Kv", type=int, default=8, required=False,
                   help="Base variable dim value (e.g., out/in channels for conv; K or J for matmul)")
    p.add_argument("--Kc", type=int, default=4, required=False,
                   help="Base kernel dim (e.g., KERNEL_DIM)")
    p.add_argument("--Vmin", type=int, default=3, required=False,
                   help="Minimum value for variable dims")
    p.add_argument("--Vmax", type=int, default=20, required=False,
                   help="Maximum value for variable dims")
    p.add_argument("--Vmin_c", type=int, default=3, required=False,
                   help="Minimum value for controlled (kernel) dim")
    p.add_argument("--Vmax_c", type=int, default=8, required=False,
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
    p.add_argument("--kernels", type=str, default="all",
                   help="Comma-separated subset to generate: matmul,conv,dwconv or 'all'")

    # Output
    p.add_argument("--output", type=str, default="mappings_output.yaml", help="YAML output path")
    return p


def parse_kernels_arg(kernels_arg: str) -> List[str]:
    if kernels_arg.strip().lower() == "all":
        return ["matmul", "conv", "dwconv"]
    parts = [p.strip().lower() for p in kernels_arg.split(",") if p.strip()]
    valid = {"matmul", "conv", "dwconv"}
    filtered = [p for p in parts if p in valid]
    if not filtered:
        raise ValueError(f"--kernels must be 'all' or a comma list of {sorted(valid)}")
    return filtered

# ---------- YAML helpers (DROP-IN REPLACEMENT) ----------
from typing import Optional

class FlowList(list):
    """Force PyYAML to dump this list in flow style: [a, b, c]."""
    pass

def _flowlist_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

# Register for both dumpers
yaml.add_representer(FlowList, _flowlist_representer)
yaml.SafeDumper.add_representer(FlowList, _flowlist_representer)

def _pyify(obj):
    """Recursively convert numpy scalars/arrays & tuples to plain Python types/lists."""
    # Avoid importing numpy unless present
    np_types = ()
    try:
        import numpy as np
        np_types = (np.generic,)
    except Exception:
        pass

    if isinstance(obj, dict):
        return { _pyify(k): _pyify(v) for k, v in obj.items() }
    if isinstance(obj, (list, FlowList)):
        return [ _pyify(v) for v in obj ]
    if isinstance(obj, tuple):
        return [ _pyify(v) for v in obj ]  # always list in YAML
    if np_types and isinstance(obj, np_types):
        # Cast numpy scalar to builtin
        return obj.item()
    return obj  # ints/str/None/etc are fine

def as_flow_list(seq) -> FlowList:
    # Normalize first so items are builtin types
    norm = _pyify(seq)
    return FlowList(norm)

def normalize_strategy_for_yaml(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure exact field shapes/types and force short lists to flow style.
    condition: list of single triplets, each triplet in flow form.
    X, Y, Divisible_tiles, dim_with_2d_unroll: all flow lists.
    """
    s = _pyify(strategy)  # strip numpy/tuples etc.
    out = {}

    # condition

    # Keep required non-list fields too
    for k in ("strategy_id",):
        if k in s:
            out[k] = s[k]
    
    cond = s.get("condition", [])
    out["condition"] = [as_flow_list(c) for c in cond]

    # X/Y/Divisible_tiles/dim_with_2d_unroll
    for k in ("X", "Y", "Divisible_tiles", "dim_with_2d_unroll"):
        v = s.get(k, [])
        out[k] = as_flow_list(v)
    return out

def normalize_mapping_for_yaml(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep order: mapping_id, kernel_type, num_strategies, strategies
    and convert inner lists to flow lists.
    """
    m = _pyify(mapping)
    strategies = [normalize_strategy_for_yaml(s) for s in m.get("strategies", [])]
    return {
        "mapping_id": m.get("mapping_id"),
        "kernel_type": m.get("kernel_type"),
        "num_strategies": len(strategies),
        "strategies": strategies,
    }

def load_yaml_if_exists(path: str) -> Dict[str, Any]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, "r") as f:
        try:
            data = yaml.safe_load(f) or {}
        except Exception:
            return {}
    return data if isinstance(data, dict) else {}

def upsert_mappings_always_overwrite_device(
    existing_payload: Dict[str, Any],
    device_config: Dict[str, Any],
    new_mappings: List[Dict[str, Any]],
    kernels_to_update: Optional[List[str]],
) -> Dict[str, Any]:
    # Always overwrite deviceOption
    payload = {"deviceOption": _pyify(device_config)}
    sa_list = existing_payload.get("SA_mappings", [])
    if not isinstance(sa_list, list):
        sa_list = []

    # Keep any existing mappings whose kernel_type is NOT being updated
    keep = []
    updating = set(kernels_to_update) if kernels_to_update else None
    for m in sa_list:
        kt = (m or {}).get("kernel_type")
        if updating is None or kt not in updating:
            keep.append(m)

    # Add/replace with new normalized mappings
    new_norm = [normalize_mapping_for_yaml(m) for m in new_mappings]
    payload["SA_mappings"] = keep + new_norm
    return payload

def write_yaml_fixed_format(path: str, payload: Dict[str, Any]) -> None:
    out = _pyify(payload)
    # enforce normalization on SA_mappings for consistent format
    out["SA_mappings"] = [normalize_mapping_for_yaml(m) for m in out.get("SA_mappings", [])]
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

def update_yaml_file(
    output_path: str,
    device_config: Dict[str, Any],
    new_mappings: List[Dict[str, Any]],
    kernels_to_update: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load output YAML (if any), overwrite deviceOption, and update only the specified
    kernels (or all if kernels_to_update is None).
    """
    existing = load_yaml_if_exists(output_path)

    # Filter new mappings if a subset is requested
    if kernels_to_update:
        kset = set(kernels_to_update)
        new_mappings = [m for m in new_mappings if m.get("kernel_type") in kset]

    merged = upsert_mappings_always_overwrite_device(
        existing_payload=existing,
        device_config=device_config,
        new_mappings=new_mappings,
        kernels_to_update=kernels_to_update,
    )
    write_yaml_fixed_format(output_path, merged)
    return merged

# ========= Top-level glue that your main() calls =========
def generate_mappings(args, kernels: List[str]) -> Dict[str, Any]:
    """
    Collect mappings (your find_* functions), then update the YAML file.
    Always overwrites deviceOption.
    """
    device_config = {
        "SystolicArrayDataflow": "WS",
        "SystolicArrayDimension": args.SA_dim,
        "deviceType": "SA",
    }

    new_mappings: List[Dict[str, Any]] = []
    if "matmul" in kernels:
        new_mappings.append(find_all_mappings_matmul(Kv=args.Kv, Vmin=args.Vmin, Vmax=args.Vmax, args=args))
    if "dwconv" in kernels:
        new_mappings.append(find_all_mappings_dwconv(Kv=args.Kv, Kc=args.Kc,
                                                     Vmin_c=args.Vmin_c, Vmax_c=args.Vmax_c,
                                                     Vmin=args.Vmin, Vmax=args.Vmax, args=args))
    if "conv" in kernels:
        new_mappings.append(find_all_mappings_conv(Kv=args.Kv, Kc=args.Kc,
                                                   Vmin_c=args.Vmin_c, Vmax_c=args.Vmax_c,
                                                   Vmin=args.Vmin, Vmax=args.Vmax, args=args))

    # Always overwrite device config; update only the kernels you asked for
    final_payload = update_yaml_file(
        output_path=args.output,
        device_config=device_config,
        new_mappings=new_mappings,
        kernels_to_update=kernels,
    )
    return final_payload

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    kernels = parse_kernels_arg(args.kernels)
    payload = generate_mappings(args, kernels)
    # merge_yaml already wrote to args.output
    print(f"Wrote/merged mappings to {args.output} for kernels: {', '.join(kernels)}")



if __name__ == "__main__":
    main()
