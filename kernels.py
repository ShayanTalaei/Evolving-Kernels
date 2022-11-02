import numpy as np
import math

def compute_kernel(X_train, X_test, name, **kwargs):
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    if name == "rbf":
        var = kwargs.get("var", 20)
        X_norm = np.sum(X_train ** 2, axis = -1)
        K = np.exp(- (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(X_train, X_train.T))/(2*var**2))
        
        X_norm = X_norm.reshape(n_train, -1).repeat(n_test, axis=1)
        XT_norm = np.sum(X_test ** 2, axis = -1).reshape(n_test, -1).repeat(n_train, axis=1)
        KT = np.exp(- (XT_norm + X_norm.T - 2 * np.dot(X_test, X_train.T))/(2*var**2))
    elif name == "linear":
        K = np.dot(X_train, X_train.T)
        KT = np.dot(X_test, X_train.T)
    elif name == "ntk":
        dataset_name = kwargs.get("dataset", "CIFAR2")
        n_layers = kwargs.get("n_layers", 2)
        if n_layers == 2:
            K = NTK2(X_train.T, X_train.T)
            KT = NTK2(X_test.T, X_train.T)
        else:
            # For multilayer networks, read it from the disk
            K = np.load("NTK_Kernels/" + 'Train_NTK_%d_layers_%d_%s.npy'%(0, n_layers, dataset_name))
            KT = np.load("NTK_Kernels/" + 'Test_NTK_%d_layers_%d_%s.npy'%(0, n_layers, dataset_name))
    print(K.shape, KT.shape)
    assert K.shape[0] == n_train and K.shape[1] == n_train
    assert K.dtype == np.float32
    assert KT.shape[0] == n_test and KT.shape[1] == n_train
    assert KT.dtype == np.float32
    return K, KT

def NTK2(X, Z):
    """This function computes NTK kernel for two-layer ReLU neural networks via
    an analytic formula.
    Input:
    X: d times n_1 matrix, where d is the feature dimension and n_i are # obs.
    Z: d times n_2 matrix, where d is the feature dimension and n_i are # obs.
    output:
    C: The kernel matrix of size n_1 times n_2.
    """
    pi = math.pi
    assert X.shape[0] == Z.shape[0]
    # X is sized d \times n
    nx = np.linalg.norm(X, axis=0, keepdims=True)
    nx = nx.T    
    nz = np.linalg.norm(Z, axis=0, keepdims=True)    

    C = np.dot(X.T, Z) #n_1 * n_2
    C = np.multiply(C, (nx ** -1))
    C = np.multiply(C, (nz ** -1))
    # Fixing numerical mistakes
    C = np.minimum(C, 1.0)
    C = np.maximum(C, -1.0)

    C = np.multiply(1.0 - np.arccos(C) / pi, C) + np.sqrt(1 - np.power(C, 2)) / (2 * pi)
    C = np.multiply(nx, np.multiply(C, nz))
    return C


# treshold = 0.9
# print("The trainig set:")
# print(f"The number of false-positive samps above {treshold} is {np.sum(np.logical_and(yhat>treshold, IKM.Y_train < 0))}.")
# print(f"The number of false-negative samps below {-treshold} is {np.sum(np.logical_and(yhat<-treshold, IKM.Y_train > 0))}.")
# print("The test set:")
# print(f"The number of false-positive samps above {treshold} is {np.sum(np.logical_and(preds>treshold, IKM.Y_test < 0))}.")
# print(f"The number of false-negative samps below {-treshold} is {np.sum(np.logical_and(preds<-treshold, IKM.Y_test > 0))}.")

# yhat = ((abs(yhat) > treshold)*np.sign(yhat)).astype(np.float32)
# preds = ((abs(preds) > treshold)*np.sign(preds)).astype(np.float32)

# K = np.outer(yhat.flatten(), yhat.flatten())
# KT = np.outer(preds.flatten(), yhat.flatten())

# IKM.Ks.append(K)
# IKM.KTs.append(KT)