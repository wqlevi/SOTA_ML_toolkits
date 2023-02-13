import torch 
import torch.nn as nn
import numpy as np
import seaborn 
import matplotlib.pyplot as plt
from img2dist import MultiVariateGaussian


param = {'batch_size':512,
        'latent_dim':512,
        'lr':5e-4,
        'iters':2000,
        'model':'Large' # 'Large' or 'Small'
        }

dataset = MultiVariateGaussian(batch_size = param['batch_size'])
sample = dataset.makeIt()

def plot_fn(target, sample, memo=""):
    bg_color = seaborn.color_palette('Greens', n_colors=256)[0]
    plt.subplot(1,2,1)
    ax = seaborn.kdeplot(x = sample[:,0],
                         y = sample[:,1],
                         fill=True,
                         cmap='Greens',
                         n_levels=20
                         )
    ax.set_title("Generated")
    ax.set_box_aspect(1)
    plt.subplot(1,2,2)
    ax1 = seaborn.kdeplot(x = target[:,0],
                          y = target[:,1],
                          fill = True,
                          cmap='Greens',
                          n_levels=20
                          )
    ax1.set_title("GT")
    ax1.set_box_aspect(1)
    plt.savefig(f"results_{memo}.png")

def noise_sampler(batch_size):
    """
    Generate Gaussian noise with shape of [batch ,2]
    """
    return np.random.normal(size=[batch_size, 2]).astype('float32')

"""
Define data and targets
"""
sample_x = torch.tensor(noise_sampler(param['batch_size']), dtype=torch.float32).cuda()[:,None,:]
target_y = sample.cuda()[:,None,:].float()

"""
Define model and optimizer
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
if param['model'] == 'Small':
    net = lambda: nn.Sequential(nn.Linear(2,128),
                            nn.ReLU(),
                            nn.Linear(128,128),
                            nn.ReLU(),
                            nn.Linear(128,256),
                            nn.ReLU(),
                            nn.Linear(256,128),
                            nn.Tanh(),
                            nn.Linear(128,128),
                            nn.Tanh(),
                            nn.Linear(128,2),
                            nn.BatchNorm1d(1)
                            )
model = net().cuda()
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad==True], lr=param['lr'], weight_decay=1e-8)

"""
Define loss
"""
KL = nn.KLDivLoss(reduction='batchmean', log_target=True) 
MSE = nn.MSELoss()

"""
trainings
"""
loss_ls = []
for i in range(param['iters']):
    # sample_x: grad=False
    predict_y = model(sample_x)
    # predict_y: grad=True, target_y: grad=False
    #loss = .5*KL(predict_y,target_y)+MSE(predict_y, target_y)
    loss = MSE(predict_y, target_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"loss:{loss:.3f}")
    with torch.no_grad():
        loss_ls.append(loss.item()) 

memo = param['model']+'Net_'+str(param['iters'])
plot_fn(target_y.squeeze().detach().cpu(),predict_y.squeeze().detach().cpu(), memo=memo)
plt.plot(loss_ls)
plt.show()
