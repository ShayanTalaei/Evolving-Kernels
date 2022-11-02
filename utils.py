import numpy as np
import torch
from torch import nn



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
        preds. """bookboobookk
        if true_labels.shape[1] == 1:
            mid = np.mean(true_labels)
            n = len(true_labels)
            true_labels = true_labels.reshape((n, 1))
            preds = preds.reshape((n, 1))
            preds = preds > mid
            inds = true_labels > mid
            return np.mean(preds == inds)
        groundTruth = np.argmax(true_labels, axis=1).astype(np.int32)
        predictions = np.argmax(preds, axis=1).astype(np.int32)
        return np.mean(groundTruth == predictions)

def normalize(yhat, preds):
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