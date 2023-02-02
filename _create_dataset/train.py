from genericpath import sameopenfile
from random import sample
import torch
from img2dist import *
from torch import nn

import matplotlib.pyplot as plt

"""
log in training data
"""
data = MakeDist("/home/qiwang/Binary_coins.png")
data_2d = data.scatter_pix()# need to swap axis 0 and 1

# new dataset here:
data_2d = MakeSpiral()
data_2d = data_2d()
data_2d_ts = torch.tensor(data_2d,dtype=torch.float32)[:,None,:] # 1023x2

data_2d_ts_norm = (data_2d_ts - data_2d_ts.mean())/data_2d_ts.std()


src = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))

"""
initialize network, optimizer, and loss
"""
                        
net = lambda: nn.Sequential(nn.Linear(2,16), 
                            nn.LeakyReLU(),
                            nn.Linear(16,64), 
                            nn.LeakyReLU(),
                            nn.Linear(64,128), 
                            nn.LeakyReLU(),
                            nn.Linear(128,256), 
                            nn.LeakyReLU(),
                            nn.BatchNorm1d(1),
                            nn.Linear(256,512), 
                            nn.LeakyReLU(),
                            nn.Linear(512,256), 
                            nn.LeakyReLU(),
                            nn.Linear(256,128), 
                            nn.LeakyReLU(),
                            nn.Linear(128,64), 
                            nn.LeakyReLU(),
                            nn.Linear(64,2),
                            nn.BatchNorm1d(1)
                            #nn.Sigmoid()
                            )

model = net().cuda()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad==True], lr=5e-4, weight_decay=1e-8)

# [BUG] The KL could be incorrectly implemented
KL = nn.KLDivLoss(reduction='batchmean', log_target=True) 
MSE = nn.MSELoss()

"""
get original and training dataset
"""
# random Gaussian 
sample_x = src.sample((data_2d.shape[0],1)).cuda()
# original dist.
target_y = data_2d_ts_norm.cuda()

def plot_fn(pred, origin):
    plt.scatter(pred[:,0],pred[:,1])
    plt.scatter(origin[:,0],origin[:,1],color='r',alpha=.5)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()

"""
training loop
"""
loss_ls = []
for i in range(20000):
    # sample_x: grad=False
    predict_y = model(sample_x)
    # predict_y: grad=True, target_y: grad=False
    loss = .5*KL(predict_y,target_y)+MSE(predict_y, target_y)
    #loss = MSE(predict_y, target_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{loss:.3f}")
    with torch.no_grad():
        loss_ls.append(loss.item()) 
    #if not i%1000:
plot_fn(predict_y.squeeze().detach().cpu(),
        target_y.squeeze().detach().cpu())
plt.plot(loss_ls)

