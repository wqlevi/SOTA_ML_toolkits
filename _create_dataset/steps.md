# Goal
A repository of making toy datasets for more generative models to be validated. Including a baseline model of 2-layered linear network(2x256, leakyReLU, 256x2, Tanh).

## make dataset
[Toy discrete distributions](./_create_dataset/img2dist.py)
* target dataset of scatter 1D distribution of image    
* Some spiral dataset  

[Dataset from torch](https://pytorch.org/docs/stable/distributions.html)
* original dataset of 1D multivariate Gaussian  
  
    
## model
### Autoencoder
1. 2k iterations
 - model of linear layers (2->16->64->128->256->512, 512->256->128->64->2)  
![result Large model](figs/results_LargeNet_2000.png)
 - smaller network(2->128->128->256->128->128->2)  
![result Small model](figs/results_SmallNet_2000.png)
2. 20k iterations
![result large model 20k](figs/results_LargeNet_20000.png)
![result small model 20k](figs/results_SmallNet_20000.png)
 - Neural ODE
### GAN 
 - with same level of parameterization?


## Technical problems
1. Should the input tensor require grad? Otherwise it does not update?
    * depends whether the inputs are expected to be updated(e.g. score-matching function that updates the diffusion tensors)
2. Is KL divergence better than MSE?
    * KL divergence easily faces vanishing gradient(remember to put inputs in log space)  
    ![KL result, loss=-0.879](KL.png)

    * MSE tends to move to convergence, but very slow  
    ![MSE result, loss=0.007](MSE.png)

    * Mixing the above two  
    ![MIX result, loss=0.067](mix.png)
3. Additional force to push two distribution to overlap?
    * It's observed from applying negative log-likelihood(NLL) and KL divergence seperatedly, that only applying NLL to measure the reconstruction loss tends to generate more fuzzy distribution than KL, which forces the details of the distribution to match.

4. Autoencoder seems already work better than a GAN, given same level of parameterization?
