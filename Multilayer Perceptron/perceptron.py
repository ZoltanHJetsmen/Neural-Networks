import numpy as np
import pandas as pd
import random
from collections import namedtuple

def readDataset(filename):

    data = pd.read_csv(filename, index_col=False, sep=" ")
    
    y = data.iloc[:,len(data.columns)-1]
    y = np.array(y)
    X = data.iloc[:,0:len(data.columns)-1]
    X = np.array(X)

    # Embaralha os dados
    indices = np.random.choice(len(X), len(X), replace=False)
    X_values = X[indices]
    y_values = y[indices]

    dataset = namedtuple('datset', 'X Y')

    d = dataset(X=X_values, Y=y_values)

    return d

#Definição da função tangente hiperbólica
def tanh(x):

    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

# Função que realiza o forward propagation, onde é gerado o y previsto de acordo com 
# a entrada e os pesos
def forward(x, weights):

    # geração da função net: w1x1 + w2x2 + ... + wnxn + theta
    net = np.matmul(x,weights[0:x.shape[1]]) + weights[-1]
    # Aplicação da função tangente hiperbólica
    f_net = tanh(net)
    
    return f_net, net

# Função que realiza a backward propagation, onde ocorre a atualização dos pesos de acordo as derivas do erro
def backward(x, y, f_net, net, weights, alpha):

    error = y - f_net
    # Soma dos erros de cada exemplo (linha)
    sum_errors = sum(error*error)
    # Derivada da função tangente hiperbólica
    D_f_net = 4/(np.power(np.exp(-net)+np.exp(net),2))
    # Derivada do erro em relação a w
    D_error_w = np.matmul(-2*error*D_f_net,x)
    # Atualização dos pesos
    n_weights = weights[0:x.shape[1]] -alpha*D_error_w
    # Derivada do erro em relação a theta
    D_error_t = -2*error*D_f_net
    # Atuzalição do theta
    n_theta = weights[-1] - alpha*D_error_t  

    return np.append(n_weights, n_theta), sum_errors

def perceptron(dataset, test, threshold, alpha):

    weights = []

    # Leitura dos pesos com uma distribuição normal indo de -1 até 1, onde o theta está na ultima coluna
    for i in range(dataset.X.shape[1] + 1):
        weights.append(random.uniform(-1, 1))

    error = 2*threshold

    while(error > threshold):
        # Forward
        f_net, net = forward(dataset.X, weights)
        # Backward
        weights, error = backward(dataset.X, dataset.Y, f_net, net, weights, alpha)

        print("Erro: " + str(error))

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    print("Saída prevista")
    print(forward(test.X, weights)[0])
    print("Saída esperada")
    print(test.Y)
    
    

perceptron(readDataset("train.txt"), readDataset("test.txt"), 0.001, 0.1)