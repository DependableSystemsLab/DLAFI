def write_matmul_params(i, k, j, header_path="DLAFI_MatMul.h"):
    with open(header_path, "w") as f:
        f.write(f"#define MAT_DIM_I {i}\n")
        f.write(f"#define MAT_DIM_K {k}\n")
        f.write(f"#define MAT_DIM_J {j}\n")

def write_conv_params(i, k, j, header_path="DLAFI_Conv.h"):
    with open(header_path, "w") as f:
        f.write(f"#define OUT_CHANNELS {i}\n")
        f.write(f"#define IN_CHANNELS {k}\n")
        f.write(f"#define KERNEL_DIM {j}\n")

def write_dwconv_params(i, j, header_path="DLAFI_DWConv.h"):
    with open(header_path, "w") as f:
        f.write(f"#define OUT_CHANNELS {i}\n")
        f.write(f"#define KERNEL_DIM {j}\n")


write_matmul_params(12, 12, 10)