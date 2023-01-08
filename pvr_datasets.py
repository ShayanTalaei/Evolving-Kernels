import numpy as np
import torch

def pointer_label(vect, index=3):
    pointer = 0
    for i in range(index):
        pointer = 2*pointer + vect[i]
    return 2*vect[pointer + index]-1

def pointer_majority(vect, index=3, win_size=3):
    pointer = 0
    for i in range(index):
        pointer = 2*pointer + vect[i]
    data = vect[index:]
    win = data[np.arange(pointer, pointer+win_size)% (2**index)]
    return 1 if np.sum(win)>1 else -1

def create_boolean_pvr(label_function, index=3):
    bits = index + 2**index
    nums = np.arange(0, 2**bits)
    bin_nums = ((nums.reshape(-1,1) & (np.flip(2**np.arange(bits)))) != 0).astype(int)
    labels = np.array(list(map(label_function, bin_nums)))
    return bin_nums, labels

######################################################################################
### PVR functions ###

def f1(x):
    # f(x)= (1-x1)/2 maj(x3,x4,x5) + (1+x1)/2 maj(x6,x7,x8)
    # maj (a,b,c)=1/2(a+b+c-abc)
    
    def maj(a, b, c):
        return 1/2*(a+b+c-a*b*c)
    
    y = (1-x[:,0])/2*maj(x[:,2], x[:,3], x[:,4]) + (1+x[:,0])/2*maj(x[:,5], x[:,6], x[:,7])
    return y

def f0(x):
    # f(x) = x0
    
    y = x[:, 0]
    return y

def f01(x):
    # f(x) = 1/2(x0 + x1)
    
    y = 1/2*(x[:, 0] + x[:, 1])
    return y

def f02(x):
    # f(x) = x0(1 + x1)
    
    y = x[:, 0]*(1 + x[:, 1])
    return y

def f03(x):
    # f(x) = maj(x0, x1, x2)
    
    def maj(a, b, c):
        return 1/2*(a+b+c-a*b*c)
    y = maj(x[:, 0], x[:, 1], x[:, 2])
    return y

def f2(x):
    # f(x) = x0 + x1 + x1x2 + x1x2x3
    
    y = x[:, 0] + x[:, 1] + x[:, 1]*x[:, 2] + x[:, 1]*x[:, 2]*x[:, 3]
    return y
######################################################################################
### To create dataset ###

# f = f1
# f_name = "f1"

# for d in [10, 20]:
#     for n in [100, 500, 2000, 5000]:
#         X = np.random.choice([-1, +1], size=(n, d))
#         Y = f(X)
#         title = f"data/pvr/{f_name}_n={n}_d={d}"
#         torch.save((X, Y), title)
#         print(f"Dataset {title} is saved!")



# X, Y = create_boolean_pvr(pointer_majority, 3)
# torch.save((X, Y), "data/pvr/pointer_majority_3_3")
# X, Y = torch.load("data/pvr/pointer_majority_3_3")
