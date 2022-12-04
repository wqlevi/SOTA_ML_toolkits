import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
import numpy as np

class Lorenz(torch.nn.Module):
    """
    The object of the functon has to be a nn.Module with 2 input args,
    here  it's a chaotic Lorenz:
    where coefficient are 10, 28, -8/3, for dx,dy,dz respectively
    """
    def __init__(self):
        super(Lorenz, self).__init__()
        self.lin = torch.nn.Linear(5, 3, bias=True)
        W = torch.tensor([[-10, 10, 0, 0, 0],
                          [28, -1, 0, -1, 0],
                          [0, 0, -8/3, 0, 1]])
        self.lin.weight = torch.nn.Parameter(W)

    def forward(self, t, x):
        y = torch.ones([1, 5]).to(device)
        y[0][0] = x[0][0]
        y[0][1] = x[0][1]
        y[0][2] = x[0][2]
        y[0][3] = x[0][0] * x[0][2]
        y[0][4] = x[0][0] * x[0][1]
        return self.lin(y)

def vis_3d(input):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1,1,1, projection='3d')
    if isinstance(input[0],np.ndarray):
        z = np.array([o for o in input])
    else:
        z = np.array([o.detach().numpy() for o in input])
    z = np.reshape(z, [-1,3])
    for i in range(len(z)):
        ax.plot(z[i:i+10,0], z[i:i+10,1], z[i:i+10,2], color = plt.cm.jet(i/len(z)/1.6))

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show()

def main():
    fn = Lorenz().to(device)
    with torch.no_grad():
        # odeint(func_dy_dt, y_0, delta_t), with the 1st arg as the derivative of function
        true_lorenz = odeint(fn, true_y0, t, method='rk4')
    return true_lorenz

def Lorenz_fn(state,t):
    '''
    Characterize the first derivatives of Lorenz attractor
    Output: dx, dy, dz
    '''
    x,y,z = state
    return 10*(y-x), x*(28-z) - y, x*y-(8/3)*z



if __name__ == '__main__':
    data_size = 1000 # number of samples
    device = "cpu"
    true_y0 = torch.tensor([[-8, 7, 27]],dtype=torch.float32).to(device) # intial value 
    t = torch.linspace(0, 10, data_size).to(device)
    main()
    vis_3d(true_lorenz)


