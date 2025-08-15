import re
from analyse_waveform import analyse
DIM = 32
def parse_vcd(filename):
    # Regex patterns to capture relevant lines
    module_pattern = re.compile(r'( *)\$scope module mesh_(\d+)_(\d+) \$end')
    # signal_pattern = re.compile(r'( *)\$var wire 32 (\S+) (c[12]) \[31:0\] \$end')
    signal_pattern = re.compile(r'( *)\$var wire( +)(\d+)( +)(\S+)( +)(c[12] \[|io_in_a \[|io_in_control_propagate)(.*)\$end')
    time_pattern = re.compile(r'#(\d+)')
    value_pattern = re.compile(r'b([01]+) (\S*)')

    hash_array = [ [ ['0','0','0','0'] for i in range(DIM+3) ]for j in range(DIM+3)]
    values = [ [ [ [] for k in range(4)] for i in range(DIM+3) ]for j in range(DIM+3)]
    hash_reverse = {}

    current_time = None
    cnt = 0
    with open(filename, 'r') as file:
        for line in file:
            
            if cnt % 1000000==1:
                print("--------------------------------",line)
                # break
            # b00000000000000000000000000001100 w<Q
            cnt += 1
            line = line.strip()
            if current_time == None:
                # Check for module definitions
                module_match = module_pattern.match(line)
                if module_match:
                    _,x, y = module_match.groups()
                    current_module = [int(x), int(y)]  # Remember current module position
                    continue
                
                # Check for signal definitions within the module
                signal_match = signal_pattern.match(line)
                if signal_match:
                    print(signal_match.groups())
                    _, _,_,_, hash_code,_, signal_type,_ = signal_match.groups()
                    idx = {'c1 [': 0, 'c2 [': 1, 'io_in_control_propagate': 2, 'io_in_a [': 3}[signal_type]
                    hash_array[current_module[0]][current_module[1]][idx] = hash_code
                    hash_reverse[hash_code] = [current_module[0], current_module[1], idx]
                    continue
            
            # Capture time points
            time_match = time_pattern.match(line)
            if time_match:
                current_time = time_match.group(1)
                continue

            if current_time!=None and len(line)<=4:
                value = int (line[0])
                hash_tag = line[1:]
                if hash_tag in hash_reverse:
                    x,y,c = hash_reverse[hash_tag]
                    if(c>=2):
                        values[x][y][c].append(value)
                        print(x,y,c,"!!",value)
                        # exit()
                
            # Capture signal values
            value_match = value_pattern.match(line)
            if value_match:
                value, hash_tag = value_match.groups()
                # print(line)
                value = int(value,2)
                if hash_tag in hash_reverse:
                    # print(value)
                    print(hash_tag,value)
                    x,y,c = hash_reverse[hash_tag]
                    if(c>=2):
                        values[x][y][c].append(value)
                        print(x,y,c,"!",value)
                        # exit()
                    else:
                        print("line ",cnt)
                        if(c == 0 and value != 0):
                            propagate = values[x][y][2][-1]
                            input = values[x][y][3][-1]
                            if(input != 1 or propagate != 0):
                                continue
                            values[x][y][c].append(value)
                        if(c == 1 and value != 0):
                            propagate = values[x][y][2][-1]
                            input = values[x][y][3][-1]
                            if(input != 1  or propagate != 1):
                                continue
                            values[x][y][c].append(value)

    return values

# Usage
filename = 'waveforms/waveform.vcd'
result = parse_vcd(filename)
import pickle

shapewf = ['32conv',18,10,3,3]
resultfilename = 'wf_result'
for x in shapewf:
    resultfilename += str(x) + "-"
resultfilename  += '.pkl'

with open(resultfilename, 'wb') as file:  # Use binary mode ('wb')
    pickle.dump(result, file)

analyse(shapewf)