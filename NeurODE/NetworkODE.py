import numpy as np
import torch,sys
import matplotlib.pyplot as plt
from tqdm import tqdm
"""
The same function as was in Euler's method:
    y(t) = exp(-t/5) * sin(t)
"""

def sigmoid(x):
    return 1/(1+torch.exp(-x))
def sigmoid_grad(x):
    return sigmoid(x) * (1-sigmoid(x))
def neural_net(x,weight,bias):
    """
    Two linear layer with non-linear gate:
    first layer = sigmoid(x*W0+b0)
    second layer = first_layer * W1 + b1
    """
    s_z = sigmoid(torch.matmul(x,weight[0]) + bias[0])
    return torch.matmul(s_z,weight[1])+bias[1]
def dNNdx(x,weight):
    """
    Derivative of neural network over x
    """
    s_z_grad = sigmoid_grad(torch.matmul(x,weight[0]) + bias[0])
    mul = torch.mul(weight[0].T,weight[1])
    return torch.matmul(s_z_grad,mul)

def loss(x):
    """
    A L2 loss of derivative of predicted function to it orignal function derivative 
    """
    x.requires_grad = True
    psi = psi_t(x) # the y_pred
    ddN = dNNdx(x,weight)
    psi_t_x = neural_net(x,weight,bias) + x*ddN
    return torch.mean((psi_t_x - f(x,psi))**2)
def plot_dst(x,psi_trial,y):
    fig,ax = plt.subplots()
    with torch.no_grad():
        ax.plot(x,y.numpy(),'k',label='analytical solution')
        ax.plot(x,psi_trial.numpy(),'g--',label='NN prediction')
        plt.legend()
        plt.show()

def train(epochs,lr=0.01):
    for i in tqdm(range(epochs)):
        loss_ = loss(x)
        loss_.backward()
        weight[0].data -= lr*weight[0].grad.data
        weight[1].data -= lr*weight[1].grad.data
        bias[0].data   -= lr*bias[0].grad.data
        bias[1].data   -= lr*bias[1].grad.data

        weight[0].grad.zero_()
        weight[1].grad.zero_()
        bias[0].grad.zero_()
        bias[1].grad.zero_()

        print("loss :",loss_.item())

if __name__ == '__main__':
    epochs = int(sys.argv[1])
    lr = float(sys.argv[2])
    weight = [torch.randn((1,10), requires_grad = True), torch.randn((10,1),requires_grad = True)]
    bias = [torch.randn(10,requires_grad = True), torch.randn(1,requires_grad = True)]

    A = 0 # initial condition
    psi_t = lambda x:A + x*neural_net(x,weight,bias)  # the predicted y
    f = lambda x,psi: torch.exp(-x/5.0) * torch.cos(x) - psi/5.0 # this psi should be predicted y


    x = torch.unsqueeze(torch.linspace(0,5,100),dim=1)
    y = torch.exp(-(x/5.0))*torch.sin(x) # original y

    train(epochs, lr)
    psi_trial = psi_t(x)
    plot_dst(x,psi_trial,y)
