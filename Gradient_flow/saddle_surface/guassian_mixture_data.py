import numpy as np
def make_guassian():
    radius = 10
    num_mixture = 8
    theta = np.linspace(0,2*np.pi, num_mixture, endpoint=False)
    xs = radius * np.sin(theta, dtype=np.float32)
    ys = radius * np.cos(theta, dtype=np.float32)
    centers = np.vstack([xs,ys]).T

    sample = np.random.normal(0,1,2)
    sample = sample / (np.sqrt(sample[1]**2 + sample[0]**2))
    return sample
