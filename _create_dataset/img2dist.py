from PIL import Image
from torch import scalar_tensor
import torch
import numpy as np

class MakeDist:
    """
    transform image to binary distribution
    """
    def __init__(self,path:str, width=64):
        self.path = path
        self.width = width
    
    def __call__(self):
        return self.scatter_pix()

    def scatter_pix(self):
        # transform image into binary values as scatter plot
        w = self.width
        data = Image.open(self.path).resize((w,w)).convert("L")
        pix = data.load()
        black_pix = [(x,y) for x in range(w) for y in range(w) if pix[x,y]!=0]

        return ([t[0] for t in black_pix], [w - t[1] for t in black_pix])

    def flatten_data(x,y)->list:
        # flatten list of x,y coordinates to 1D list
        return [*x,*y]

    def unflatten_data(data:list)->tuple:
        # unflatten 1D list back to 2D x,y 
        half_len_data = len(data)//2
        return (data[:half_len_data], data[half_len_data:])


class MakeSpiral:
    """
    Enter number of datapoints, return 2 spiral as tuple
    Output:
        [2*batch_size, 2]
    """
    def __init__(self, N=400):
        self.N = N
    def makeit(self):
        theta = np.sqrt(np.random.rand(self.N))*2*np.pi
        ra = 2*theta + np.pi
        rb = -ra
        data_a = np.array([ra*np.cos(theta), ra*np.sin(theta)]).T
        data_b = np.array([rb*np.cos(theta), rb*np.sin(theta)]).T
        xa = data_a + np.random.randn(self.N,2)
        xb = data_b + np.random.randn(self.N,2)
        #return xa,xb
        return torch.tensor(np.concatenate((xa,xb)))

class MultiGaussian_GT:
    """
    Only serves as the labels of Gaussian
    """
    def __init__(self, batch_size=512, N_modality=8):
        self.batch_size = batch_size
        self.modality = N_modality
    def makeit(self):
        theta = np.linspace(0, 2*np.pi, self.modality)
        xs, ys = 1.0*np.sin(theta), 1.0*np.cos(theta) 
        return torch.tensor([xs,ys]).T
class MultiVariateGaussian:
    def __init__(self, batch_size = 512, N_modality=8):
        self.batch_size = batch_size
        self.modality = N_modality
        self.std = 0.02
    def makeCenter(self):
        theta = np.linspace(0, 2*np.pi, self.modality)
        xs, ys = 1.0*np.sin(theta), 1.0*np.cos(theta) 
        return xs,ys
    def makeIt(self):
        ls = []
        i_center = np.random.choice(self.modality, self.batch_size)
        xs,ys = self.makeCenter()
        center = np.concatenate([xs[:,None], ys[:,None]], 1)
        sample_points = np.random.normal(loc=center[i_center, :], scale=self.std)
        ls.append(sample_points)
        return torch.tensor(np.array(ls)).squeeze()


