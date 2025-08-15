
import pickle 
import torch
result = None
DIM=32
def analyse(shapewf):
    resultfilename = 'wf_result'
    for x in shapewf:
        resultfilename += str(x) + "-"
    resultfilename  += '.pkl'
    with open(resultfilename, 'rb') as file:  # Use binary mode ('rb')
        result = pickle.load(file)
        
    print(result)
        
    shape = []
    for x in range(1,len(shapewf)):
        shape.append(shapewf[x])
    shape = tuple(shape)
    print(shape)
    max_v = 1
    for i in shape:
        max_v *= i
    # print(max_v)
    w_x = torch.ones(shape) * -1
    w_y = torch.ones(shape) * -1
    for j in range(DIM):
        for i in range(DIM):
            print("----")   
            print(i,j)
            print(result[i][j][0])
            print(result[i][j][1])
            for k in range(2):
                for x in result[i][j][k]:
                    if (x >  max_v):
                        continue
                    offset = x
                    offset -= 1
                    index = [0 for ii in range(len(shape))]
                    for id in range(len(shape) - 1, -1, -1):
                        index[id] = offset%shape[id]
                        offset = offset // shape[id]
                    print(x,index)
                    w_x[tuple(index)] = i
                    w_y[tuple(index)] = j
            
    # print(result)
    print("-----------------------------------------------------")
    print("w_x: ",w_x)
    print("w_y: ",w_y)

    def get_axis(w_x,w_y, d):
        diff = set()
        indices = (w_x > -1).nonzero(as_tuple=False)
        # print("INDICES ",indices)
        for index in indices:
            if index[d] + 1 != w_x.shape[d]:
                x1 = w_x[tuple(index)]
                y1 = w_y[tuple(index)]
                index2 = list(index)
                index2[d] += 1
                x2 = w_x[tuple(index2)]
                y2 = w_y[tuple(index2)]
                if x1 == -1 or x2 == -1:
                    continue
                diff.add(((x2-x1).item(),(y2-y1).item()))
        return diff
                
    two_p = -1
    two_d = -1
    szinner = -1
    x_axis = []
    y_axis = []  
    hint_diff = {
    0: {(1.0, 0.0), (-15.0, 5.0), (-15.0,-10.0) },
    1: {(0.0, 1.0)},
    2: {(0.0, 0.0)},
    3: {(0.0, 0.0)}}  
    for d in range(len(shape)):
        diff = get_axis(w_x,w_y,d)
        # diff = hint_diff[d]
        in_x = 0
        in_y = 0
        sx = 0
        sy = 0
        tx = 0
        ty = 0
        tmp_d = -1
        for p in diff:
            a,b = p
            if(a != 0):
                in_x = 1
                if a > 0:
                    sx = a
                if a < 0 :
                    tx = -a
            else:
                tmp_d = 0
            if(b != 0):
                in_y = 1
                if b > 0:
                    sy = b
                if b < 0 :
                    ty = -b
            else:
                tmp_d = 1
        if in_y and in_x:
            two_d = tmp_d
            two_p = d
            if two_d == 1:
                szinner = (sx+tx)//sx
            else :
                szinner = (sy+ty)//sy
            x_axis.append([sx,tx,len(shape)])
            y_axis.append([sy,ty,len(shape)+1])
        else:
            if in_x:
                x_axis.append([sx,tx,d])
            if in_y:
                y_axis.append([sy,ty,d])
        print(d,diff)
        
    print("--------------")
    x_axis.sort()
    y_axis.sort()
    print(x_axis)
    print(y_axis)
    # 0 {(0.0, 1.0), (0.0, -15.0)}
    # 1 {(1.0, 0.0), (-15.0, 0.0)}
    # 2 {(0.0, 0.0)}
    # 3 {(0.0, 0.0)}
    def get_mapping(ind):
        print(ind)
        if two_p!= -1:
            z = ind[two_p]
            if(two_d == 1):
                xx = z%szinner
                yy = z//szinner
            else:
                xx = z//szinner
                yy = z%szinner
            
            ind.append(xx)
            ind.append(yy) 
            print(xx,yy,"**")
            
        macX = 0
        macY = 0
        
        for xx in x_axis:
            s,t,d = xx
            macX += s*ind[d]
            if(t>0):
                macX %= t+s
        for yy in y_axis:
            s,t,d = yy
            macY += s*ind[d]
            if(t>0):
                macY %= t+s
        return macX, macY

    print(get_mapping([0,0,0,0]))
    print(get_mapping([10,0,0,0]))
    print(get_mapping([0,4,0,0]))
    print(get_mapping([10,4,0,0]))
    print(get_mapping([20,3,0,0]))

    print(get_mapping([45,3,0,0]))

if __name__ == "__main__":
    analyse(['conv',10,7,5,5])
    
    
# 0 {(0.0, 1.0), (-10.0, -15.0), (5.0, -15.0)}
# 1 {(1.0, 0.0)}
# 2 {(0.0, 0.0)}
# 3 {(0.0, 0.0)}
# --------------
# [[1.0, 0, 1], [5.0, 10.0, 4]]
# [[1.0, 15.0, 5]]
# [0, 0, 0, 0]
# 0.0 0.0 **
# (0.0, 0.0)
# [10, 0, 0, 0]
# 0.0 10.0 **
# (0.0, 10.0)
# [0, 4, 0, 0]
# 0.0 0.0 **
# (4.0, 0.0)
# [10, 4, 0, 0]
# 0.0 10.0 **
# (4.0, 10.0)
# [20, 3, 0, 0]
# 1.0 4.0 **
# (8.0, 4.0)
# [60, 3, 0, 0]
# 3.0 12.0 **
# (3.0, 12.0)