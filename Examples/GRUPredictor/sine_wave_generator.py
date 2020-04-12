import numpy as np
import torch

np.random.seed(2)


# period
T = 20
# length of sequence
L = 200
# number of samples
N = 1000

x = np.empty((N, L), 'int64')
x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
data = np.sin(x / 1.0 / T).astype('float32')
torch.save(data, open('traindata.pt', 'wb'))