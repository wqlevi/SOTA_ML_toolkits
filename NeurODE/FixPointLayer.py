"""
The fix point finding algorithm:
    find z* such that: z* = tanh(Wz* + x)
"""
import torch 
import torch.nn as nn
from modules import FixPointLayer

layer = FixPointLayer(50).to("cuda:0")
device = torch.device("cuda:0")
X = torch.randn(10,50, device = device)
Z = layer(X)
print(f"after {layer.iteration} iterations, with error {layer.err}")


