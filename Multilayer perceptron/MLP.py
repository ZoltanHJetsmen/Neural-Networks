import numpy as np
import pandas as pd
import random
from collections import namedtuple
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def readDataset(filename):

    data = pd.read_csv(filename, index_col=False, header=None)

    y = data.iloc[:,len(data.columns)-1]
    y = np.array(y)
    X = data.iloc[:,0:len(data.columns)-1]
    X = np.array(X)

    # Shuffle Data
    indices = np.random.choice(len(X), len(X), replace=False)
    X_values = X[indices]
    y_values = y[indices]

    dataset = namedtuple('datset', 'X Y')

    d = dataset(X=X_values, Y=y_values)

    return d

def processing(dataset):

    #x = preprocessing.normalize(dataset.X, norm='l2')
    scaler = StandardScaler()
    scaler.fit(dataset.X)
    x = scaler.transform(dataset.X)


    onehot_encoder = OneHotEncoder(sparse=False)
    y = dataset.Y.reshape(len(dataset.Y), 1)
    y = onehot_encoder.fit_transform(y)
    
    lenght = dataset.X.shape[0]

    x_train = x[0:int(0.7*lenght), :]
    y_train = y[0:int(0.7*lenght), :]

    x_test = x[int(0.7*lenght):, :]
    y_test = y[0:int(0.7*lenght), :]
        
    dataset = namedtuple('datset', 'X Y')

    train = dataset(X=x_train, Y=y_train)

    test = dataset(X=x_test, Y=y_test)

    return train, test

# Definição da função sigmoid
def sigmoid(x):

    return 1/(1+np.exp(-x))

def mlp_forward(x, hidden_weights, output_weights):

    net = np.matmul(x,hidden_weights[:,0:len(x)].transpose()) + hidden_weights[:,-1]
    f_net_h = sigmoid(net)

    net = np.matmul(f_net_h,output_weights[:,0:len(f_net_h)].transpose()) + output_weights[:,-1]
    f_net_o = sigmoid(net)

    return f_net_o, f_net_h

def mlp_backward(dataset, i, hidden_weights, output_weights, f_net_o, f_net_h, alpha, hidden_units, n_classes):

    x = dataset.X[i,:]
    y = dataset.Y[i,:]

    error = y - f_net_o

    #delta da output layer
    delta_o = error*f_net_o*(1-f_net_o)

    #delta da hidden layer
    w_o = output_weights[: ,0:hidden_units]
    
    delta_h = (f_net_h*(1-f_net_h))*(np.matmul(delta_o, w_o))

    x_aux = np.concatenate((x,np.ones(1)))[np.newaxis, :]
    delta_h = delta_h[:, np.newaxis]

    delta_o = delta_o[:, np.newaxis]
    f_net_aux = np.concatenate((f_net_h,np.ones(1)))[np.newaxis, :]
    
    #print(delta_o)
    #print(np.concatenate((f_net_h,np.ones(1))))

    output_weights = output_weights - -2*alpha*np.matmul(delta_o, f_net_aux)

    hidden_weights = hidden_weights - -2*alpha*np.matmul(delta_h, x_aux)    

    error = sum(error*error)

    return hidden_weights, output_weights, error 

def MLP(dataset, hidden_units, n_classes, threshold, alpha):

    train, test = processing(dataset)

    print(train.X)

    hidden_weights = np.zeros((hidden_units, dataset.X.shape[1] + 1))
    output_weights = np.zeros((n_classes, hidden_units+1))

    for i in range(hidden_units):
        for j in range(dataset.X.shape[1] + 1):
            hidden_weights[i][j] = random.uniform(-1, 1)

    for i in range(n_classes):
        for j in range(hidden_units+1):
            output_weights[i][j] = random.uniform(-1, 1)

    sum_errors = 2* threshold
    epoch = 0
    while(sum_errors > threshold):
        sum_errors = 0
        for i in range(train.X.shape[0]):
            # Forward
            f_net_o, f_net_h = mlp_forward(train.X[i, :], hidden_weights, output_weights)

            # Backward hidden_weights, output_weights, error = 
            hidden_weights, output_weights, error = mlp_backward(train, i, hidden_weights, output_weights, f_net_o, f_net_h, alpha, hidden_units, n_classes)

            sum_errors += error
        epoch += 1
        
        if(epoch % 100 == 0):
            print(sum_errors)

    print(dataset.X)

    for i in range(4):
        y, q = mlp_forward(dataset.X[i,:], hidden_weights, output_weights)
        print(dataset.X[i,:])
        print(y)

MLP(readDataset("wine.csv"), 2, 3, 0.01, 0.1)