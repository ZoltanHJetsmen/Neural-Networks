import numpy as np
import pandas as pd
import random
from collections import namedtuple
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#Function that reads and randomize the dataset, returning a namedtuple -> dataset.X and dataset.Y
def readDataset(filename, y_collumns):

    data = pd.read_csv(filename, index_col=False, header=None)
    
    y = data.iloc[:,len(data.columns)-y_collumns: len(data.columns)]
    y = np.array(y)
    X = data.iloc[:,0:len(data.columns)-y_collumns]
    X = np.array(X)

    indices = np.random.choice(len(X), len(X), replace=False)
    X_values = X[indices]
    y_values = y[indices]

    dataset = namedtuple('datset', 'X Y')

    d = dataset(X=X_values, Y=y_values)

    return d

# Function that transform divides the dataset in train and test and transform the Y values in binary, OneHotEnconding
def processing(dataset, percentage, method):

    if(method == "C"):
        scaler = StandardScaler()
        scaler.fit(dataset.X)
        x = scaler.transform(dataset.X)
        onehot_encoder = OneHotEncoder(sparse=False)
        y = dataset.Y.reshape(len(dataset.Y), 1)
        y = onehot_encoder.fit_transform(y)
    else:
        x = dataset.X
        min_max_scaler = preprocessing.MinMaxScaler()
        y = min_max_scaler.fit_transform(dataset.Y)

    lenght = dataset.X.shape[0]

    x_train = x[0:int(percentage*lenght), :]
    y_train = y[0:int(percentage*lenght), :]

    x_test = x[int(percentage*lenght):, :]
    y_test = y[int(percentage*lenght):, :]
        
    dataset = namedtuple('datset', 'X Y')

    train = dataset(X=x_train, Y=y_train)

    test = dataset(X=x_test, Y=y_test)

    return train, test

# Sigmoid function
def sigmoid(x):

    return 1/(1+np.exp(-x))

# Forward 
def mlp_forward(x, hidden_weights, output_weights,method):

    f_net_h = []

    for i in range(len(hidden_weights)):
        if i == 0:
            net = np.matmul(x,hidden_weights[i][:,0:len(x)].transpose()) + hidden_weights[i][:,-1]
            f_net = sigmoid(net)
        else:
            net = np.matmul(f_net_h[i-1],hidden_weights[i][:,0:len(f_net_h[i-1])].transpose()) + hidden_weights[i][:,-1]
            f_net = sigmoid(net)
        
        f_net_h.append(f_net) 

    net = np.matmul(f_net_h[len(f_net_h)-1],output_weights[:,0:len(f_net_h[len(f_net_h)-1])].transpose()) + output_weights[:,-1]
        
    f_net_o = sigmoid(net)
    
    return f_net_o, f_net_h

# Backward
def mlp_backward(dataset, j, hidden_weights, output_weights, f_net_o, f_net_h, alpha, hidden_units, n_classes, method):

    x = dataset.X[j,:]
    y = dataset.Y[j,:]

    error = y - f_net_o

    delta_o = error*f_net_o*(1-f_net_o)
    
    delta_h = []

    for i in range(len(hidden_units)-1, -1, -1):

        if(i == len(hidden_units)-1):
            w_o = output_weights[: ,0:hidden_units[i]]
            delta = (f_net_h[i]*(1-f_net_h[i]))*(np.matmul(delta_o, w_o))
        else:
            w_o = hidden_weights[i+1][:,0:hidden_units[i]]
            delta = (f_net_h[i]*(1-f_net_h[i]))*(np.matmul(delta, w_o))

        delta_h.insert(0,delta)
        
    delta_o = delta_o[:, np.newaxis]
    f_net_aux = np.concatenate((f_net_h[len(hidden_units)-1],np.ones(1)))[np.newaxis, :]
    output_weights = output_weights - -2*alpha*np.matmul(delta_o, f_net_aux)
    
    for i in range(len(hidden_units)-1, -1, -1):
        delta = delta_h[i][:, np.newaxis]
        f_net_aux = np.concatenate((f_net_h[i],np.ones(1)))[np.newaxis, :]    

        if i == 0:
            x_aux = np.concatenate((x,np.ones(1)))[np.newaxis, :]
            hidden_weights[i] = hidden_weights[i] - -2*alpha*np.matmul(delta, x_aux)
        else:
            f_net_aux = np.concatenate((f_net_h[i-1],np.ones(1)))[np.newaxis, :]
            hidden_weights[i] = hidden_weights[i] - -2*alpha*np.matmul(delta, f_net_aux)

    error = sum(error*error)

    return hidden_weights, output_weights, error 

def testing(train, test, hidden_weights, output_weights, method):

    if(method == "C"):
        counter = 0

        for i in range(test.X.shape[0]):
            y_hat, q = mlp_forward(test.X[i,:], hidden_weights, output_weights, method)
            y_hat = np.argmax(y_hat)
            y = np.argmax(test.Y[i,:])
            if y == y_hat:
                counter += 1

        print("Accuracy: " + str(counter/test.X.shape[0]))
    
    else:
        sum_errors = 0
        
        for i in range(train.X.shape[0]):
            y_hat, q = mlp_forward(train.X[i,:], hidden_weights, output_weights, method)
            if(i < 10):
                print(train.Y[i,:])
                print(y_hat)
                print()
            error = train.Y[i,:] - y_hat
            error = error*error
            sum_errors += sum(error)
        
        print("Squared mean error: " + str(sum_errors/test.X.shape[0]))
        

def MLP(dataset, hidden_layers ,hidden_units, n_classes, threshold, alpha, method):

    if(len(hidden_units) != hidden_layers):
        print("The parameter hidden_units must have its length the same value that hidden_layers.")
        return

    if(method != "R" and method != "C"):
        print("The parameter method must be R (Regression) or C (Classification).")
        return
    
    train, test = processing(dataset, 0.7, method)

    hidden_weights = []

    for i in range(hidden_layers):
        if(i == 0):
            aux = np.zeros((hidden_units[i], dataset.X.shape[1] + 1))
        else:
            aux = np.zeros((hidden_units[i], hidden_units[i-1] + 1))

        hidden_weights.append(aux)
    
    for i in range(hidden_layers):
        for j in range(hidden_units[i]):
            if(i == 0):
                for k in range(dataset.X.shape[1] + 1):
                    hidden_weights[i][j][k] = random.uniform(-1, 1)
            else:
                for k in range(hidden_units[i-1]+1):
                    hidden_weights[i][j][k] = random.uniform(-1, 1)

    output_weights = np.zeros((n_classes, hidden_units[len(hidden_units)-1]+1))

    for i in range(n_classes):
        for j in range(hidden_units[hidden_layers-1]+1):
            output_weights[i][j] = random.uniform(-1, 1)

    sum_errors = 2* threshold
    epoch = 0
    while(sum_errors > threshold):
        sum_errors = 0
        for i in range(train.X.shape[0]):
            # Forward
            f_net_o, f_net_h = mlp_forward(train.X[i, :], hidden_weights, output_weights, method)
            
            # Backward hidden_weights, output_weights, error = 
            hidden_weights, output_weights, error = mlp_backward(train, i, hidden_weights, output_weights, f_net_o, f_net_h, alpha, hidden_units, n_classes, method)

            sum_errors += error
    
        epoch += 1

        if method == "R":
            sum_errors = sum_errors/train.X.shape[0]
        
        if(epoch % 100 == 0):
            print(sum_errors)


    testing(train, test, hidden_weights, output_weights, method)

#MLP(readDataset("wine.csv", 1), 1, [2] , 3, 0.1, 0.1, "C")

MLP(readDataset("default_features_1059_tracks.txt", 2), 1, [30], 2, 0.003, 0.1, "R")

