
import re
import sys

def parse_vcd_weight(chipyard_dir, dim=16):
    """
    Parse a VCD file and extract the 'weight' signal values for each mesh_x_y tile.

    Returns:
        values: a dim x dim list of lists, where values[x][y] is the list of weight values for mesh_x_y.
    """
    module_pattern = re.compile(r'( *)\$scope module mesh_(\d+)_(\d+) \$end')
    signal_pattern = re.compile(r'( *)\$var wire\s+\d+\s+(\S+)\s+weight\s+\[\d+:\d+\] \$end')
    value_pattern = re.compile(r'b([01]+) (\S+)')

    values = [[[] for _ in range(dim)] for _ in range(dim)]
    hash_to_xy = {}

    current_module = None
    filename = f"{chipyard_dir}/generators/gemmini/waveforms/waveform.vcd"
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Find module scope
            module_match = module_pattern.match(line)
            if module_match:
                _, x, y = module_match.groups()
                current_module = (int(x), int(y))
                continue

            # Find weight signal in current module
            signal_match = signal_pattern.match(line)
            if signal_match and current_module is not None:
                hash_code = signal_match.group(2)
                x, y = current_module
                hash_to_xy[hash_code] = (x, y)
                continue

            # Parse value changes
            value_match = value_pattern.match(line)
            if value_match:
                value_bin, hash_code = value_match.groups()
                if hash_code in hash_to_xy:
                    x, y = hash_to_xy[hash_code]
                    value_int = int(value_bin, 2)
                    values[x][y].append(value_int)

    return values

def print_weight_values(values):
    dim = len(values)
    for x in range(dim):
        for y in range(dim):
            print(f"mesh_{x}_{y} weight values: {values[x][y]}")

def convert_flat_to_multidim(values, SAdim=16):

    max_value  = 1
    for i in dim_array:
        max_value *= i
    w_x = -np.ones(dim_array)
    w_y = -np.ones(dim_array)
    for j in range(SAdim):
        for i in range(SAdim):
            for x in values[i][j]:
                if x > max_value:
                    continue
                offset = x - 1
                index = [0 for _ in range(len(dim_array))]
                for id in range(len(dim_array) - 1, -1, -1):
                    index[id] = offset % dim_array[id]
                    offset //= dim_array[id]
                w_x[tuple(index)] = i
                w_y[tuple(index)] = j
    
    return w_x, w_y

def get_axis(w_x, w_y, target_dim):
    """
    Analyze the waveform data to find the axis of the mesh.
    """
    diff = set()
    indices = (w_x > -1).nonzero(as_tuple=False)
    for index in indices:
        if index[target_dim] + 1 != w_x.shape[target_dim]:
            x1 = w_x[tuple(index)]
            y1 = w_y[tuple(index)]
            index2 = list(index)
            index2[target_dim] += 1
            x2 = w_x[tuple(index2)]
            y2 = w_y[tuple(index2)]
            if x1 == -1 or x2 == -1:
                continue
            diff.add(((x2-x1).item(),(y2-y1).item()))
    return diff
def analyze_waveform(diff, dim_array, SAdim=16):

    X = []
    Y = []
    Div_Tiles = []
    d_2unrolls = None
    for d in range(len(dim_array)):
        axis_diff = get_axis(w_x, w_y, d)
        # print(f"Axis {d} differences: {axis_diff}")
        xd = 0
        yd = 0
        txd = 0
        tyd = 0
        two_D = None
        for a,b in axis_diff:
            if a == 0 and b == 0:
                # no tiling over this dimension
                continue
            if a > 0 and b == 0:
                xd = a
            if a == 0 and b > 0:
                yd = b
            if a > 0 and b > 0:
                print(f"Unexpected axis difference: {a}, {b} at dimension {d}")
                exit(1)
        for a,b in axis_diff:
            if a < 0 and b == 0:
                txd = -a + xd  
            if a == 0 and b < 0:
                tyd = -b + yd  
        for a,b in axis_diff:
            if a < 0 and b > 0:
                two_D = 'x'
                txd = -a + xd
            if a > 0 and b < 0:
                two_D = 'y'
                tyd = -b + yd
        for a,b in axis_diff:
            if a < 0 and b < 0:
                if two_D == 'x':
                    tyd = -b + yd
                if two_D == 'y':
                    txd = -a + xd
        X.append((xd, d, txd))
        Y.append((yd, d, tyd))
        if two_D is not None:
            d_2unrolls = (d, two_D)
    X = sorted(X, key=lambda f: f[0])
    Y = sorted(Y, key=lambda f: f[0])
    if d_2unrolls is not None:
        d, two_d = d_2unrolls
        Div_Tiles = [d]
    else:
        if len(X) > 0:
            if(X[-1][2] != SAdim - (SAdim % X[-1][0]))
            Div_Tiles.append(X[-1][1])
        if len(Y) > 0:
            if(Y[-1][2] != SAdim - (SAdim % Y[-1][0]))
            Div_Tiles.append(Y[-1][1])
    X = [f[0] for f in X]
    Y = [f[0] for f in Y]
    return X, Y, Div_Tiles, d_2unrolls

def main():
    if len(sys.argv) < 2:
        print("Usage: python read_waveform.py <chipyard_dir> [dim]")
        sys.exit(1)

    chipyard_dir = sys.argv[1]
    dim = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    values = parse_vcd_weight(chipyard_dir, dim=dim)
    print_weight_values(values)

if __name__ == "__main__":
    main()