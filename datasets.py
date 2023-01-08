import numpy as np
import dataget
import sys
import pickle as pickle
import time
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.fftpack import dct, idct
from skimage.color import rgb2gray
import random
sys.path.append('./linearized_neural_networks')
from preprocess import *

from sklearn.model_selection import train_test_split
import torch

ROOT = "./data"

DATASETS = ["CIFAR10", "mnsit", "FMNIST2", "CIFAR2", "synth", "CIFAR3", "pointer_3", "pointer_majority_3_3", "pointer_majority_3_6"]

HYPER_PARAMETERS = {}
HYPER_PARAMETERS["CIFAR10"] = {"num_classes": 10, "mean": 0.1, "expand": True}
HYPER_PARAMETERS["CIFAR2"] = {"num_classes": 2, "mean": 0.0, "expand": False}
HYPER_PARAMETERS["CIFAR3"] = {"num_classes": 3, "mean": 0.333, "expand": True}
HYPER_PARAMETERS["mnist"] = {"num_classes": 10, "mean": 0.1, "expand": True}
HYPER_PARAMETERS["FMNIST2"] = {"num_classes": 2, "mean": 0.5, "expand": False}
HYPER_PARAMETERS["pointer_3"] = {"num_classes": 2, "mean": 0.0, "expand": False}
HYPER_PARAMETERS["pointer_majority_3_3"] = {"num_classes": 2, "mean": 0.0, "expand": False}
HYPER_PARAMETERS["pointer_majority_3_6"] = {"num_classes": 2, "mean": 0.0, "expand": False}

for d in [10, 20]:
    for n in [100, 500, 2000, 5000]:
        dataset_name = f"f1_n={n}_d={d}"
        DATASETS.append(dataset_name)
        HYPER_PARAMETERS[dataset_name] = {"num_classes": 2, "mean": 0.0, "expand": False}
for d in [2, 4, 10]:
    for n in [100, 500, 2000]:
        dataset_name = f"f0_n={n}_d={d}"
        DATASETS.append(dataset_name)
        HYPER_PARAMETERS[dataset_name] = {"num_classes": 2, "mean": 0.0, "expand": False}

def load_dataset(dataset_name, **kwargs):
    if dataset_name == "FMNIST2":
        labels = kwargs.get("labels", [2, 9])
        return load_fmnist(labels)
    elif dataset_name == "CIFAR2":
        labels = kwargs.get("labels", [3, 0])
        ratio = kwargs.get("ratio", 1)
        return load_cifar2(labels, ratio)
    elif ("pointer" in dataset_name) or (dataset_name[0] == "f"):
        return load_pvr(dataset_name, **kwargs)
    else:
        print("Need to prepare dataset!")
        (X_train, Y_train, X_test, Y_test) = prep_data(dataset=dataset_name, CNN=False, noise_index=0)
    
    return (X_train, Y_train, X_test, Y_test)

def load_fmnist(labels):
    X_train, y_train, X_test, y_test = dataget.image.fashion_mnist().get()
    ind = [i for i in range(len(y_train)) if y_train[i] in labels]
    X_train, y_train = X_train[ind], y_train[ind]
    n = len(ind)
    X_train = X_train.reshape(n, -1).astype(np.float32)/255.0
    y_train = y_train.reshape((n, 1))
#     pdb.set_trace()
    for i in range(len(y_train)):
        y_train[i] = labels.index(y_train[i])+1
    
    ind = [i for i in range(len(y_test)) if y_test[i] in labels]
    X_test, y_test = X_test[ind], y_test[ind]
    n = len(ind)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32)/255.0
    y_test = y_test.reshape((n, 1))
    for i in range(len(y_test)):
        y_test[i] = labels.index(y_test[i])+1
    return (X_train, y_train.astype(np.float32), X_test, y_test.astype(np.float32))
    
def unpickle(add):
    with open(add, 'rb') as fo:
        val = pickle.load(fo, encoding='latin1')
    return val

def cifarTrain():
    """This function reads the CIFAR-10 training examples from the disk."""
    X = []
    Y = []
    for i in range(1, 6):
        data = unpickle(ROOT + '/cifar_10_py/data_batch_%d'%(i))
        #unpickle('./datasets/cifar10py/data_batch_%d'%(i))
        tempX = data['data']
        n = tempX.shape[0]
        tempX = tempX.reshape((n, 3, 32, 32))
        tempX = tempX.transpose((0, 2, 3, 1))
        X.append(tempX)

        tempy = np.array(data['labels'])
        tempy = tempy.reshape((len(tempy), 1))
        Y.append(tempy)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return (Y, X)

def cifarTest():
    """This function reads the CIFAR-10 test examples from the disk."""
    data = unpickle(ROOT + '/cifar_10_py/test_batch')
    tempX = data['data']
    n = tempX.shape[0]
    tempX = tempX.reshape((n, 3, 32, 32))
    tempX = tempX.transpose((0, 2, 3, 1))

    tempy = np.array(data['labels'])
    tempy = tempy.reshape((len(tempy), 1))
    return (tempy, tempX)

def cifar_input(mode, greyscale, CNN):
    """This function preprocesses the CIFAR-10 data. 
    Inputs:
        mode: 'train' or 'test'
        greyscale: boolean. If true the images are converted to greyscale.
        CNN: boolean. If false, the images are flattened. """
    if mode == 'train':
        Y, X = cifarTrain()
    else:
        Y, X = cifarTest()        

    n = len(Y)
    n1 = 32
    n2 = 32
    nc = 3
    if greyscale:
        nc = 1    
    Z = np.zeros((n, n1, n2, nc))
    d = n1 * n2 * nc
    for i in range(n):
        image = X[i]
        if greyscale:
            image = rgb2gray(image)
        image = image - np.mean(image)    
        image = image / np.linalg.norm(image.flatten()) * np.sqrt(d)    
        if greyscale:
            Z[i, :, :, 0] = image
        else:
            Z[i] = image
    return (Y, Z)

def load_cifar2(labels, ratio):
    assert len(labels) == 2
    Y, X = cifar_input('train', True, False)
    inds = [i for i in range(len(Y)) if Y[i, 0] in labels]
    X, Y = X[inds, :], Y[inds, :]
    comb = list(zip(X, Y))
    random.shuffle(comb)
    X, Y = zip(*comb)
    X, Y = np.array(X), np.array(Y)
    inds = []
    samples_per_class = int(5000*ratio)
    chosen_samps = [0]*2
    for i in range(len(Y)):
        ind = labels.index(Y[i, 0])
        if chosen_samps[ind] < samples_per_class:
            inds.append(i)
            Y[i, 0] = 2*ind - 1
            chosen_samps[ind] += 1
        if sum(chosen_samps) == samples_per_class*2:
            break
    X, Y = X[inds].astype(np.float32), Y[inds].astype(np.float32)
    YT, XT = cifar_input('test', True, False)
    # Choosing cats and airplanes
    inds = [i for i in range(len(YT)) if YT[i, 0] in labels]
    XT, YT = XT[inds, :], YT[inds, :]
    for i in range(len(YT)):
        ind = labels.index(int(YT[i, 0]))
        YT[i, 0] = (2*ind - 1)
    return X, Y, XT.astype(np.float32), YT.astype(np.float32)

def load_pvr(name, **kwargs):
    address = f"data/pvr/{name}"
    train_size = kwargs.get("train_size", 0.8)
    shuffle = kwargs.get("shuffle", True)
    random_state = kwargs.get("random_state", 0)
    (X, y) = torch.load(address)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, train_size=train_size, shuffle=shuffle)
    return X_train.astype(np.float32), y_train.astype(np.float32).reshape(-1, 1), X_test.astype(np.float32), y_test.astype(np.float32).reshape(-1, 1)
    