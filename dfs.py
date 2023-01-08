import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
# from tqdm import tqdm
import os 

import pdb

from trainer import *
from utils import *

def log(new_results, **kwargs):
    file_path = setup["file_path"]
    results = torch.load(file_path)
    print(len(list(new_results.keys())))
    results.update(new_results)
    print(len(results.keys()))
    torch.save(results, file_path)
    
def choose_kernel(IKM, iter_number, report, **kwargs):
    for kernel, params in setup[f"kernel {iter_number}"]:
        if kernel == "rbf":
            new_report = report + "-> rbf"
        elif kernel == "linear":
            new_report = report + "-> linear"
        for key, val in params.items():
            new_report += f", {key}: {str(val)}"
        if iter_number == 1:
            IKM.set_Ds()
            IKM.make_kernel_matrices(0, kernel, **params)
            cross_validate(IKM, iter_number, new_report, **kwargs)
        else:
            choose_g(IKM, iter_number, new_report, kernel, **kwargs)

def choose_g(IKM, iter_number, report, kernel, **kwargs):
    yhat, preds = kwargs.get("yhat"), kwargs.get("preds")
    for g in setup[f"g {iter_number}"]:
        if g == "identity":
            new_report = report + "-> g identity"
        elif g == "normalize":
            new_report = report + "-> g normalize"
            ## Normalize the predictions
            yhat, preds = normalize_preds(yhat, preds)
        IKM.set_Ds(yhat=yhat, preds=preds)
        IKM.make_kernel_matrices(ind=iter_number, kernel=kernel)
        mix_kernels(IKM, iter_number, new_report, **kwargs)
        
def mix_kernels(IKM, iter_number, report, **kwargs):
    if iter_number > 1:
        for weights in setup[f"mixing {iter_number}"]:
            new_report = report + f"-> mix ({', '.join(list(map(lambda x: str(x), weights)))})"
            IKM.combine_kernels(weights)
            cross_validate(IKM, iter_number, new_report, **kwargs)
            
def cross_validate(IKM, iter_number, report, **kwargs):
    IKM_copy = deepcopy(IKM)
    avg_diag = IKM.avg_diag_of_kernel()
    verbose_level = kwargs.get("verbose_level", 0)
    verbose = iter_number <= verbose_level
    for log_reg_ratio in setup[f"log_regs {iter_number}"]:
        reg = avg_diag * (10**log_reg_ratio)
        new_report = report + f"-> reg {reg:.3f}"
        # if new_report not in logs:
        if verbose:
            print(new_report)
        yhat, preds, res, Theta = IKM.solve(reg, verbose=verbose)
        if verbose:
            print(f"______________________________________")
        # log(new_report, res, **kwargs)
        results = kwargs["new_results"]
        results[new_report] = res
        kwargs["yhat"], kwargs["preds"] = yhat, preds
        IKM_copy = deepcopy(IKM)
        perform_iteration(IKM, iter_number+1, new_report, **kwargs)
        IKM = IKM_copy
        
def perform_iteration(IKM, iter_number, report, **kwargs):
    max_iter = setup["max iterations"]
    if iter_number <= max_iter:
        if iter_number <= kwargs.get("verbose_level", 0):
            print(f"------- level: {iter_number} -------")
        new_report = report + f" /{iter_number}: "
        if iter_number == 1:
            kwargs["new_results"] = {}
        choose_kernel(IKM, iter_number, new_report, **kwargs)
        if iter_number == 1:
            print(len(list(kwargs["new_results"].keys())))
            log(**kwargs)
            
