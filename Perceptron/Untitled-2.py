def backward():

    x = dataset.X[i,:]
    y = dataset.Y[i,:]

    error = f_net_o - y

    #delta da output layer
    delta_o = error*f_net_o*(1-f_net_o)

    #delta da hidden layer
    w_o = hidden_weights[,0:hidden_weights.shape[1]]
    delta_h = (f_net_h*(1-f_net_h))*sum(delta_o * w_o)

    output_weights = output_weights + np.matmul(alpha * delta_o, np.concatenate((f_net_h,np.ones(f_net_h.shape[0]), axis=1)))

    hidden_weights = hidden_weights + np.matmul(alpha * delta_h, np.concatenate((x,np.ones(x.shape[0]), axis=1)))    

    error = sum(error*error)



