from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# this is using GPU for training if available
# GPU training has similar performance to CPU for such small model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future = 0):
        # input has shape of (batchSize, seqLen, stateDim)
        outputs = []
        
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double, device = device)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double, device = device)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device = device)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double, device = device)

        # decompose input to seqLen number of chunks, each chunk has shape of (batchSize, stateDim)
        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs

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
seq = Sequence()
if torch.cuda.is_available():
    seq = seq.cuda()
# convert model parameter type to double
seq.double()

print(next(seq.parameters()).is_cuda)

criterion = nn.MSELoss()
# use LBFGS as optimizer since we can load the whole data to train
optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
#optimizer = optim.SGD(seq.parameters(), lr = 0.8)
#begin to train
for i in range(25):
    print('STEP: ', i)
    # see https://pytorch.org/docs/stable/optim.html for why we define a closure
    def closure():
        optimizer.zero_grad()
        out = seq(input)
        loss = criterion(out, target)
        print('loss:', loss.item())
        loss.backward()
        return loss
    optimizer.step(closure)
#    optimizer.zero_grad()
#    out = seq(input)
#    loss = criterion(out, target)
#    loss.backward()
#    optimizer.step()

    # begin to predict, no need to track gradient here
    with torch.no_grad():
        future = 1000
        pred = seq(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        print('test loss:', loss.item())
        if pred.is_cuda:
            y = pred.detach().cpu().numpy()
        else:
            y = pred.detach().numpy()
    # draw the result
    plt.figure(figsize=(30,10))
    plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    def draw(yi, color):
        plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
        plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
    draw(y[0], 'r')
    draw(y[1], 'g')
    draw(y[2], 'b')
    plt.savefig('predict%d.pdf'%i)
    plt.close()