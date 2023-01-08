import numpy as np
import torch
from torch import nn
import gspread
import pandas as pd
import itertools


import pdb

def expand(y, mean, maxVal=None):
    # Expand the labels
    n = y.shape[0]
    if maxVal == None:
        maxVal = int(y.max())
    Y = np.zeros((n, maxVal + 1), dtype=np.float32)
    Y[np.arange(n), y[:, 0].astype(np.int32)] = 1.0
    Y = Y - mean
    assert Y.dtype == np.float32
    ## mean should be set to 0 after calling this function.
    return Y, maxVal

def compute_accuracy(true_labels, preds):
        """This function computes the classification accuracy of the vector
        preds. """
        if true_labels.shape[1] == 1:
            mid = np.mean(true_labels)
            n = len(true_labels)
            true_labels = true_labels.reshape((n, 1))
            preds = preds.reshape((n, 1))
            true_preds = preds > mid
            inds = true_labels > mid
            return np.mean(np.logical_or((true_preds == inds), (abs(true_labels-preds)<0.1)))
        groundTruth = np.argmax(true_labels, axis=1).astype(np.int32)
        predictions = np.argmax(preds, axis=1).astype(np.int32)
        return np.mean(groundTruth == predictions)

def normalize_preds(yhat, preds):
    mean, std = np.mean(yhat), np.std(yhat)
    yhat = (yhat - mean)/std
    preds = (preds - mean)/std
    return yhat, preds
    
def make_model(layers_dim, **kwargs):
#     pdb.set_trace()
    layers = []
    for inp, out in zip(layers_dim[0:-1], layers_dim[1:]):
        layers.append(nn.Linear(inp, out))
        layers.append(nn.ReLU())
    model = nn.Sequential(*layers[:-1])
    lr = kwargs.get("lr", 0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return model, optimizer

def train_model(model, optimizer, X_train, y_train, epochs, **kwargs):
    batch_size = kwargs.get("batch_size", 128)
    criterion = kwargs.get("criterion", nn.MSELoss())
    batches = int(X_train.shape[0] / batch_size) + 1
    average_loss = None
    for epoch in range(epochs):
        for i in range(batches):
            Xb, yb = X_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]
            Xb, yb = Xb.reshape(Xb.shape[0], -1), yb.reshape(yb.shape[0], -1)
            Xb, yb = torch.from_numpy(Xb), torch.from_numpy(yb)
            optimizer.zero_grad()
            predictions = model(Xb)
            loss = criterion(predictions, yb)
            average_loss = average_loss*0.9 + loss*0.1 if average_loss else loss
            loss.backward()
            optimizer.step()
    return average_loss
            
def evaluate_model(model, X, y, **kwargs):
    with torch.no_grad():
        X = X.reshape(X.shape[0], -1)
        preds = model(torch.from_numpy(X))
        criterion = kwargs.get("criterion", nn.MSELoss())
        loss = criterion(preds, torch.from_numpy(y))
        acc = compute_accuracy(y, preds.numpy()) * 100
        dataset = kwargs.get("dataset", "-")
        print("{} loss is {:6.4f}.".format(dataset, loss))
        print("{} acc is {:6.4f}%.".format(dataset, acc))
    return loss, acc
    
def train_test_NN(datasets, epochs=20):
    (X_train, y_train, X_test, y_test) = datasets
    model, optimizer = make_model(layers_dim=[1024, 1000, 1])
    train_model(model, optimizer, X_train, y_train, epochs=epochs)
    train_loss, train_acc = evaluate_model(model, X_train, y_train, dataset="Train")
    test_loss, test_acc = evaluate_model(model, X_test, y_test, dataset="Test")
    res["Train error"] = train_loss
    res["Test error"] = test_loss
    res["Train accuracy"] = train_acc
    res["Test accuracy"] = test_acc
    print('Training Error is %f'%(res["Train error"]))
    print('Test Error is %f'%(res["Test error"]))
    print('Training Accuracy is %f'%(res["Train accuracy"]))
    print('Test Accuracy is %f'%(res["Test accuracy"]))
    return res

def extract_feature_2D_table(data, feature):
    table = {}
    for key1, value1 in data.items():
        row = {}
        for key2, value2 in value1.items():
            row[key2] = value2[feature]
        table[key1] = row
    return table

def update_spread_sheet(title, file_path):
    print("Updating the spread sheet.")
    logs = torch.load(file_path)
    df = pd.DataFrame(logs).transpose()
    gc = gspread.service_account(filename='iterative-learning-methods-4a3e9c5a47ae.json')
    sh = gc.open_by_url('https://docs.google.com/spreadsheets/d/1vF_0iuL4E_sfiLjcL1dePgbHShk5Qj3R0gYJeCkXpCk/edit#gid=0')
    worksheet_list = [ws.title for ws in sh.worksheets()]
    if title not in worksheet_list:
        print(f"Sheet {title} is created.")
        worksheet = sh.add_worksheet(title=title, rows=150000, cols=6)
    worksheet = sh.worksheet(title)
    df["Instructions"] = df.index
    new_df = df.filter(['Instructions', 'Train error', 'Test error', 'Train accuracy', 'Test accuracy'], axis=1)
    worksheet.update([new_df.columns.values.tolist()] + new_df.values.tolist())

def comp_dot_funcs(f, d, monomial):
    # Computes sum f.monomial for all possible values of monomial (its list of powers).
    # Input dimension of f is d.
    deg_monom = len(monomial)
    binary_combs = np.array(list(itertools.product([-1, 1], repeat=deg_monom)))
    X = np.zeros((2**deg_monom, d))
    X[:, monomial] = binary_combs
    f_X = f(X)
    f_dot_X = f_X*np.prod(X, 1)
    return np.sum(f_dot_X)

def get_boolean_representation(X_all, f_X, deg_to_show):
    d = X_all.shape[1]
    poly = {}
    powerSet = []
    for k in range(deg_to_show+1):
        powerSet += list(itertools.combinations(np.arange(d), k))
    count = len(f_X)
    for power in powerSet:
        poly[power] = np.sum(np.dot(f_X.T, np.prod(X_all[:, power],1, keepdims=1)))/count
    return poly