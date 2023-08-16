import numpy as np
i, j = np.ogrid[:3, :4]
x = 10*i+j
shape = (2,2) # window size
step_size = 2
v = np.lib.stride_tricks.sliding_window_view(x, shape)[:,::step_size]
