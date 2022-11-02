import numpy as np
import time
import math
import scipy.linalg as scl
import scipy.sparse as ss
from copy import deepcopy

from datasets import *
from kernels import *
from utils import *

class IterativeKernelModel:
    
    def __init__(self, dataset_name, **kwargs):
        
        (self.X_train, self.Y_train, self.X_test, self.Y_test) = kwargs.get("datasets", load_dataset(dataset_name, **kwargs))
        self.n_train, self.n_test = self.X_train.shape[0], self.X_test.shape[0]
        print(f"Train samples: {self.n_train}, Test samples: {self.n_test}")
        self.X_train, self.X_test = self.X_train.reshape(self.n_train, -1), self.X_test.reshape(self.n_test, -1)
        hyper_parameters = HYPER_PARAMETERS[dataset_name]
        self.mean, self.should_expand = hyper_parameters["mean"], hyper_parameters["expand"]
        if self.should_expand:
            self.Y_train, maxVal = expand(self.Y_train, self.mean, maxVal=None)
            self.Y_test, maxVal = expand(self.Y_test, self.mean, maxVal=maxVal)
            self.mean = 0.0
#         pdb.set_trace()
        self.set_Ds() 
        self.K, self.KT = None, None
        self.Ks, self.KTs = [], []
#         self.make_kernel_matrices(dataset=dataset_name, ind=0, **kwargs)
        self.logs = []
    
    def set_Ds(self, **kwargs):
        yhat, preds = kwargs.get("yhat", None), kwargs.get("preds", None)
        if "yhat" not in kwargs.keys():
            self.D_train = np.eye(self.n_train, dtype=np.float32)
            self.D_test = np.eye(self.n_test, dtype=np.float32) 
        else:
            self.D_train = np.diag(yhat.flatten(), k=0)
            self.D_test = np.diag(preds.flatten(), k=0)
    
    def make_kernel_matrices(self, ind, kernel, **kwargs):
        K, KT = compute_kernel(self.X_train, self.X_test, kernel, **kwargs)
        self.K = self.D_train @ K @ self.D_train
        self.KT = self.D_test @ KT @ self.D_train
        ## Normalize the kernels
        self.normalize_kernels()
        if ind < len(self.Ks):
            self.Ks[ind] = deepcopy(self.K)
            self.KTs[ind] = deepcopy(self.KT)
        else:
            self.Ks.append(self.K)
            self.KTs.append(self.KT)
        
    def combine_kernels(self, weights=None):
        kernels_count = len(self.Ks)
        if weights == None:
            weights = np.ones(kernels_count)
        K, KT = np.zeros_like(self.K), np.zeros_like(self.KT)
        for i in range(len(weights)):
            K += self.Ks[i] * weights[i]
            KT += self.KTs[i] * weights[i]
        self.K, self.KT = K, KT
        self.normalize_kernels()
        
    def reset_kernels(self, ind):
        self.K, self.KT = deepcopy(self.Ks[ind]), deepcopy(self.KTs[ind])
        
    def avg_diag_of_kernel(self):
        return self.K.diagonal().mean()
    
    def normalize_kernels(self):
        avg = self.avg_diag_of_kernel()
        self.K, self.KT = self.K/avg, self.KT/avg
        
    def solve(self, reg):
        K, KT, n = self.K, self.KT, self.n_train
        ytrain, ytest = self.Y_train , self.Y_test
        RK = K + reg * np.eye(n, dtype=np.float32)
        assert K.dtype == np.float32
        assert RK.dtype == np.float32
        # print('Solving kernel regression with %d observations and regularization param %f'%(n, reg))
        t1 = time.time()
        if self.should_expand:
            Theta = scl.solve(RK, ytrain, assume_a='sym')
        else:
            cg = ss.linalg.cg(RK, ytrain[:, 0], maxiter=400, atol=1e-4, tol=1e-4) # - mean
            Theta = np.copy(cg[0]).reshape((n, 1))
        t2 = time.time()
        print('iteration took %f seconds'%(t2 - t1))
        yhat = np.dot(K, Theta) #+ mean
        preds = np.dot(KT, Theta) #+ mean
        res = {}
        res["reg"] = reg
        res["Train error"] = np.linalg.norm(ytrain - yhat) ** 2 / (len(ytrain) + 0.0)
        res["Test error"] = np.linalg.norm(ytest - preds) ** 2 / (len(ytest) + 0.0)
        res["Train accuracy"] = compute_accuracy(ytrain , yhat) #-mean
        res["Test accuracy"] = compute_accuracy(ytest, preds) #-mean
        print('Training Error is %f'%(res["Train error"]))
        print('Test Error is %f'%(res["Test error"]))
        print('Training Accuracy is %f'%(res["Train accuracy"]))
        print('Test Accuracy is %f'%(res["Test accuracy"]))
        self.logs.append(res)
        return yhat, preds, res
        
    
    