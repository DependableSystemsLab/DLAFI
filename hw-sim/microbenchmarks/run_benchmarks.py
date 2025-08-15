import subprocess

# === Step 1: Configuration ===
config = [
    {"id": 0, "name": "KERNEL_DIM",     "value": 3, "type": "c"},
    {"id": 1, "name": "KERNEL_DIM",     "value": 3, "type": "c"},  # Optional: if you split kernel dims
    {"id": 2, "name": "IN_CHANNELS",    "value": 3, "type": "v"},
    {"id": 3, "name": "KERNEL_DIM",     "value": 3, "type": "c"},
    {"id": 4, "name": "OUT_CHANNELS",   "value": 3, "type": "v"},
    {"id": 5, "name": "IN_DIM",         "value": 8, "type": "v"},
    {"id": 6, "name": "BATCH_SIZE",     "value": 1, "type": "n"},
    {"id": 7, "name": "PADDING",        "value": 0, "type": "n"},
    {"id": 8, "name": "STRIDE",         "value": 1, "type": "n"},
]

# === Step 2: Generate config.h ===
with open("config.h", "w") as f:
    for key, val in config.items():
        f.write(f"#define {key} {val}\n")

print("[+] config.h written.")

# === Step 3: Compile ===
compile_cmd = [
    "gcc", "-o", "conv_exec", "main.c", "-lm",  # add other includes if needed
]
result = subprocess.run(compile_cmd, capture_output=True, text=True)
if result.returncode != 0:
    print("[-] Compilation failed:")
    print(result.stderr)
    exit(1)
else:
    print("[+] Compilation succeeded.")

# === Step 4: Run the executable ===
run_cmd = ["./conv_exec"]
result = subprocess.run(run_cmd, capture_output=True, text=True)

print("[+] Output:")
print(result.stdout)