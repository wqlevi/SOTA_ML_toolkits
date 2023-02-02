# Goal
A repository of making toy datasets for more generative models to be validated. Including a baseline model of 2-layered linear network(2x256, leakyReLU, 256x2, Tanh).

## make dataset
* target dataset of scatter 1D distribution of image
* original dataset of 1D multivariate Gaussian
    [Dataset from torch](https://pytorch.org/docs/stable/distributions.html)
    
## model
 - 2-layered linear 
 - Neural ODE
 - Convolutional layers   


## Technical problems
1. Should the input tensor require grad? Otherwise it does not update?
2. KL divergence or MSE?
    * KL divergence easily faces vanishing gradient(remember to put inputs in log space)
    ![KL result, loss=-0.879](KL.png)

    * MSE tends to move to convergence, but very slow
    ![MSE result, loss=0.007](MSE.png)

    * Mixing the above two
    ![MIX result, loss=0.067](mix.png)
3. Additional force to push two distribution to overlap?
    * It's observed from applying negative log-likelihood(NLL) and KL divergence seperatedly, that only applying NLL to measure the reconstruction loss tends to generate more fuzzy distribution than KL, which forces the details of the distribution to match.