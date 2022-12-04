#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 16:03:18 2022

@author: qiwang
"""

import numpy as np
import numpy.random as random
from numpy.linalg import svd, norm
import torch
import torch.nn.functional as F

random.seed(10)
#W = random.randn(3,3)+1
W = np.ones((64,3,3,3))


# Naive SVD
u,s,v = svd(W) 
print(f"The first SV:{s[0]:.3f}")

# Power iteration
def eign_value(a,v)->np.ndarray:
    z = a@v/v
    return z[0]
def svd_power_iteration(A)->np.ndarray:
    h,w = A.shape
    assert h==w, "Von Mises iteration works only on diagnozable Matrix!"
    v = np.ones(w)/np.sqrt(w)
    ev = eign_value(A, v)
    while True:
        Av = A@v
        v_new = Av/norm(Av)
        ev_new = eign_value(A, v_new)
        if np.abs(ev-ev_new)<.01:
            break
        v = v_new
        ev = ev_new
        print(ev_new)
    return ev_new
s_ = svd_power_iteration(W)
print(f"\nThe first SV of power iteration:{s_:.5f}")

def power_iteration(A, num_simulations: int):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    b_k = np.random.rand(A.shape[1])

    for _ in range(num_simulations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        b_k = b_k1 / b_k1_norm

    return b_k
S = power_iteration(W, 1000)
print(f"\nThe first SV of power iteration:{S[0]:.5f}")


'''
Efficient way of calculating SV, in BigGAN paper
'''
# Gram_schmit
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


# Apply num_itrs steps of the power method to estimate top N singular values.
'''
output:
    svs : singular value
'''
def power_iteration(W, u_, update=True, eps=1e-12):
    # Lists holding singular vectors and values
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        # Run one step of the power iteration
        with torch.no_grad():
            v = torch.matmul(u, W)
            # Run Gram-Schmidt to subtract components of all other singular vectors
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            # Add to the list
            vs += [v]
            # Update the other singular vector
            u = torch.matmul(v, W.t())
            # Run Gram-Schmidt to subtract components of all other singular vectors
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            # Add to the list
            us += [u]
            if update:
                u_[i][:] = u
        # Compute this singular value and add it to the list
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
        #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
    return svs, us, vs

num_iters = 1000
u_  = [torch.randn(1,64) for _ in range(num_iters)]
W_mat = torch.tensor(W).view(W.shape[0],-1)
res = power_iteration(W_mat.to(torch.float32), u_)
