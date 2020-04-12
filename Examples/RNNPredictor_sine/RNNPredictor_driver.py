from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Core.RNN import RNN

# this is using GPU for training if available
# GPU training has similar performance to CPU for such small model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# set random seed to 0
np.random.seed(0)
torch.manual_seed(0)



# load data and make training set
data = torch.load('traindata.pt')
input = torch.from_numpy(data[3:, :-1])
target = torch.from_numpy(data[3:, 1:])
test_input = torch.from_numpy(data[:3, :-1])
test_target = torch.from_numpy(data[:3, 1:])
if torch.cuda.is_available():
    input = input.cuda()
    target = target.cuda()
    test_input = test_input.cuda()
    test_target = test_target.cuda()
# build the model
seq = RNN(numLayers=2, inputDim=1, hiddenDim=64, outputDim=1, device=device)
if torch.cuda.is_available():
    seq = seq.cuda()



print(next(seq.parameters()).is_cuda)

criterion = nn.MSELoss()
# use LBFGS as optimizer since we can load the whole data to train
optimizer = optim.Adam(seq.parameters(), lr=0.0025)
#optimizer = optim.SGD(seq.parameters(), lr = 0.8)
#begin to train
for i in range(15000):
    print('STEP: ', i)
    # see https://pytorch.org/docs/stable/optim.html for why we define a closure
    idx = torch.randint(0, input.size(0), (32,))
    batchInput = input[idx]
    batchTarget = target[idx]
#    optimizer.step(closure)
    optimizer.zero_grad()
    out = seq(batchInput)
    loss = criterion(out, batchTarget)
    loss.backward()
    optimizer.step()

    def draw(yi, color):
        plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
        plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
    if i % 10 == 0:
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 200
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            if pred.is_cuda:
                y = pred.detach().cpu().numpy()
            else:
                y = pred.detach().numpy()


        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.savefig('predict%d.pdf'%i)
        plt.close()