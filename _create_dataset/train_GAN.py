import torch 
import torch.nn as nn
import numpy as np
import seaborn 
import torch.optim as optim
import matplotlib.pyplot as plt
from img2dist import MultiVariateGaussian

param = {'batch_size':512,
        'latent_dim':256,
        #'lr':5e-4,
        'd_lr':1e-4,
        'g_lr':1e-3,
        'iters':3000,
        'model':'Small' # 'Large' or 'Small'
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
def plot_scatter(target, memo=''):
    fig = plt.figure()
    plt.style.use('ggplot')
    plt.scatter(target[:,0], target[:,1], s=10, c='b', alpha=.5)
    plt.scatter(sample[:,0], sample[:,1], s=100, c='g', alpha=.5)
    plt.title(f"{memo}")
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.show()
    fig.canvas.draw()
    fig.canvas.flush_events()

def noise_sampler(batch_size, z_dim=2):
    """
    Generate Gaussian noise with shape of [batch ,2]
    """
    return np.random.normal(size=[batch_size, z_dim]).astype('float32')

"""
Define data and targets
"""
sample_x = torch.tensor(noise_sampler(param['batch_size'], param['latent_dim']), dtype=torch.float32).cuda()
target_y = sample.cuda().float()
# TODO:
# - [x] data dimension does not match 
Gnet = lambda: nn.Sequential(nn.Linear(256,128),
                             nn.Tanh(),
                             nn.Linear(128,128),
                             nn.Tanh(),
                             nn.Linear(128,2)
                            )
Dnet = lambda: nn.Sequential(nn.Linear(2,128),
                             nn.ReLU(),
                             nn.Linear(128,128),
                             nn.ReLU(),
                             nn.Linear(128,1),
                             nn.Sigmoid()
                            )
G = Gnet().cuda()
D = Dnet().cuda()

d_loss = nn.BCELoss()
g_loss = nn.BCELoss()

d_optimizer = optim.Adam(G.parameters(), lr = param['g_lr'], betas = (.5, .999)) 
g_optimizer = optim.Adam(D.parameters(), lr = param['d_lr'], betas = (.5, .999)) 

# TODO:
# - [ ] unrolled GAN with D being able to load weights after k steps of iteration, without G updatesu
# - [ ] sGAN even diverge, D loss explodes
for i in range(param['iters']):

    """
    D updates
    """
    d_optimizer.zero_grad()
    d_real = D(target_y)
    real = torch.ones_like(d_real) # check if real and fake has grad
    fake = torch.zeros_like(d_real)
    d_real_error = d_loss(d_real, real)
    sample_x = torch.tensor(noise_sampler(param['batch_size'], param['latent_dim']), dtype=torch.float32).cuda()
    with torch.no_grad():
        din_g_fake = G(sample_x)
    d_fake = D(din_g_fake)
    d_fake_error = d_loss(d_fake, fake)
    D_LOSS = d_real_error + d_fake_error
    D_LOSS.backward()
    d_optimizer.step()
    """
    G updates
    """
    g_optimizer.zero_grad()
    sample_x = torch.tensor(noise_sampler(param['batch_size'], param['latent_dim']), dtype=torch.float32).cuda()
    g_fake = G(sample_x)
    dg_fake = D(g_fake)
    g_error = g_loss(dg_fake, real)
    g_error.backward()
    g_optimizer.step()
    print(f"D loss:{D_LOSS.item():.3f}\tG loss:{g_error.item():.3f}")
    #plot_scatter(target_y.squeeze().detach().cpu())
    if not i%100:
        #plot_fn(target_y.squeeze().detach().cpu(), g_fake.squeeze().detach().cpu())
        with torch.no_grad():
            g_fake_infer = G(sample_x) 
        plot_scatter(g_fake_infer.squeeze().cpu())
    
