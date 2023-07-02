from scipy.stats import wasserstein_distance as wd
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.arange(-10 ,10 ,.001)
# create 2 1D norm dists
# with mean=0, std=2; mean=2, std=2
dist_1 = norm.pdf(x, 0, 2) 
dist_2 = norm.pdf(x, 2, 2)

def KL(P, Q):
    divergence = np.sum(np.where(P !=0, P*np.log(P/Q),0))
    return divergence
print("\u03BC=%.3f\t\u03C3=%.3f"%(dist_1.mean(), dist_1.std()))
print("\u03BC=%.3f\t\u03C3=%.3f"%(dist_2.mean(), dist_2.std()))
print("Wasserstein distance: %.3f"%(wd(dist_1, dist_2)))
print("KL: %.3f"%(KL(dist_1, dist_2)))

z = []
for i in range(0,10,1):
    dist_2_new = norm.pdf(x,i,2)
    z.append(KL(dist_1, dist_2_new))

plt.plot(np.linspace(1,10,10), z)
plt.show()
