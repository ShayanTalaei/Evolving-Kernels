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

# X, Y = create_boolean_pvr(pointer_majority, 3)
# torch.save((X, Y), "data/pvr/pointer_majority_3_3")
# X, Y = torch.load("data/pvr/pointer_majority_3_3")
