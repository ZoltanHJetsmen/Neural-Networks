import numpy as np
import pandas as pd
import random
from collections import namedtuple
from sklearn import preprocessing

def readDataset(filename):

    data = pd.read_csv(filename, index_col=False)

    y = data.iloc[:,len(data.columns)-1]
    y = np.array(y, dtype='object')
    X = data.iloc[:,0:len(data.columns)-1]
    X = np.array(X, dtype='object')

    # Shuffle Data
    indices = np.random.choice(len(X), len(X), replace=False)
    X_values = X[indices]
    y_values = y[indices]

    dataset = namedtuple('datset', 'X Y')

    d = dataset(X=X_values, Y=y_values)

    return d

def processing(dataset):

    dataset.X = preprocessing.normalize(dataset.X, norm='l2')

    x_values = dataset.X[0:int(0.7*dataset.X.shape[1]), :]
    y_values = dataset.Y[0:int(0.7*dataset.X.shape[1]), :]

    dataset = namedtuple('datset', 'X Y')

    train = dataset(X=X_values, Y=y_values)

    x_values = dataset.X[int(0.7*dataset.X.shape[1]):, :]
    y_values = dataset.Y[int(0.7*dataset.X.shape[1]):, :]

    test = dataset(X=X_values, Y=y_values)

    return train, test

# Definição da função sigmoid
def sigmoid(x):

    return 1/(1+np.exp(-x))

def mlp_forward(x, hidden_weights, output_weights):

    net = np.matmul(x,hidden_weights[:,0:2]) + hidden_weights[:,-1]
    f_net_h = sigmoid(net)

    net = np.matmul(f_net_h,output_weights[:,0:f_net_h.shape[1]].transpose()) + output_weights[:,-1]
    f_net_o = sigmoid(net)

    return f_net_o, f_net_h

def mlp_backward(dataset, i, hidden_weights, output_weights, f_net_o, f_net_h, alpha, hidden_units, n_classes):

    x = dataset.X[i,:]
    y = dataset.Y[i,:]

    error = f_net_o - y

    #delta da output layer
    delta_o = error*f_net_o*(1-f_net_o)

    #delta da hidden layer
    w_o = hidden_weights[: ,0:hidden_weights.shape[1]]
    delta_h = (f_net_h*(1-f_net_h))*sum(delta_o * w_o)

    output_weights = output_weights + np.matmul(alpha * delta_o, np.concatenate((f_net_h,np.ones(f_net_h.shape[0])), axis=1))

    hidden_weights = hidden_weights + np.matmul(alpha * delta_h, np.concatenate((x,np.ones(x.shape[0])), axis=1))    

    error = sum(error*error)

    return hidden_weights, output_weights, error 

def MLP(dataset, hidden_units, n_classes, threshold, alpha):

    train, test = processing(dataset)

    hidden_weights = np.zeros((hidden_units, dataset.X.shape[1] + 1))
    output_weights = np.zeros((n_classes, hidden_units+1))

    for i in range(hidden_units):
        for j in range(dataset.X.shape[1] + 1):
            hidden_weights[i][j] = random.uniform(-1, 1)

    for i in range(n_classes):
        for j in range(hidden_units+1):
            output_weights[i][j] = random.uniform(-1, 1)

    dataset.Y = pd.get_dummies(dataset.Y)

    sum_errors = 2* threshold
    
    while(sum_errors > threshold):
        sum_errors = 0
        for i in range(dataset.X.shape[0]):
            
            # Forward
            f_net_o, f_net_h = mlp_forward(dataset.X[i], hidden_weights, output_weights)

            # Backward hidden_weights, output_weights, error = 
            hidden_weights, output_weights, error = mlp_backward(dataset, i, hidden_weights, output_weights, f_net_o, f_net_h, alpha, hidden_units, n_classes)

            sum_errors += error
    
    
        print(sum_errors)

MLP(readDataset("wine.csv"), 2, 3, 0.01, 0.1)