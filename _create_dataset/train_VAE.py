import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
"""
Plot functions
"""
def plot_latent(model, data, num_batches = 100):
    for i,(x,y) in enumerate(data):
        z = model.module.encoder(x.cuda())
        z = z.detach().cpu().numpy()
        plt.scatter(z[:,0], z[:,1], c=y, cmap='tab10')
        if i>num_batches:
            plt.colorbar()
            break
    plt.show()

def plot_recon(model, x_lim=(-5,10), y_lim = (-10,5), n=12):
    w = 28
    img = np.zeros((n*w,n*w))
    for i,y in enumerate(np.linspace(*y_lim,n)):
        for j,x in enumerate(np.linspace(*x_lim,n)):
            z = torch.Tensor([[x,y]], device = torch.device("cuda:0"))
            x_hat = model.decoder(z)
            x_hat = x_hat.reshape(28,28).detach().cpu().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
    plt.imshow(img, extent = [*x_lim, *y_lim])
"""
Model
"""
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512,2)
    def forward(self,x):
        x = torch.flatten(x, start_dim = 1)
        x = F.relu(self.l1(x))
        return self.l2(x)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.l1 = nn.Linear(2,512)
        self.l2 = nn.Linear(512,784)
    def forward(self,z):
        z = F.relu(self.l1(z))
        z = torch.sigmoid(self.l2(z))
        return z.reshape((-1,1,28,28))
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self,x):
        z = self.encoder(x)
        return self.decoder(z)

"""
Data
"""
data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../dataset',
            transform = torchvision.transforms.ToTensor(),
            download=False),
        batch_size = 128,
        shuffle = True
        )
"""
Train
"""
def train(model, data,epoches=20):
    opt = torch.optim.Adam(model.parameters())
    for epoch in tqdm(range(epoches)):
        for x,y in data:
            x = x.cuda()
            opt.zero_grad()
            x_hat = model(x)
            loss = ((x - x_hat)**2).sum() # MSE
            loss.backward()
            opt.step()
    return model

autoencoder = Autoencoder().cuda()
autoencoder = torch.nn.DataParallel(autoencoder, device_ids=[0,1,2])
autoencoder = train(autoencoder, data)
plot_latent(autoencoder, data)
plot_recon(autoencoder.module)
