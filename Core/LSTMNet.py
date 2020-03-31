import torch
import torch.nn as nn


class LSTMNet(nn.Module):

    def __init__(self, numLstms=1, inputDim=1, hiddenDim=32, outputDim=1, device='cpu'):
        """An example MLP network for actor-critic learning. Note that the network outputs both action and value
        # Argument
        numLstms: number of LSTM layers
        inputDim: dimensionality of input feature
        hiddenDim: dimensionality of hidden feature in LSTM
        outputDim: dimensionality of output feature
        """
        super(LSTMNet, self).__init__()
        self.device = device
        self.numLSTMs = numLstms
        self.hiddenDim = hiddenDim
        self.LSTMLayers = torch.nn.ModuleList()
        for i in range(numLstms):
            if i == 0:
                LSTMLayer = nn.LSTMCell(inputDim, hiddenDim)
            else:
                LSTMLayer = nn.LSTMCell(hiddenDim, hiddenDim)
            # the following part try to initialize forget gate with value 1
            # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/2
            n = LSTMLayer.bias_hh.size(0)
            start, end = n // 4,  n // 2
            LSTMLayer.bias_hh.data[start: end].fill_(1.0)
            LSTMLayer.bias_ih.data[start: end].fill_(1.0)
            self.LSTMLayers.append(LSTMLayer)

        self.linear = nn.Linear(hiddenDim, outputDim)


    def forward(self, x, future = 0):

        outputs = []
        h_t = []
        c_t = []
        for _ in range(self.numLSTMs):
            h_t.append(torch.zeros(x.size(0), self.hiddenDim, dtype=torch.float, device = self.device))
            c_t.append(torch.zeros(x.size(0), self.hiddenDim, dtype=torch.float, device = self.device))

        for i, x_t in enumerate(x.chunk(x.size(1), dim=1)):
            for j in range(self.numLSTMs):
                if j == 0:
                    h_t[j], c_t[j] = self.LSTMLayers[j](x_t, (h_t[j], c_t[j]))
                else:
                    h_t[j], c_t[j] = self.LSTMLayers[j](h_t[j - 1], (h_t[j], c_t[j]))
            output = self.linear(h_t[-1])
            outputs.append(output)

        for i in range(future):
            for j in range(self.numLSTMs):
                if j == 0:
                    h_t[j], c_t[j] = self.LSTMLayers[j](output, (h_t[j], c_t[j])) #  use final output as input
                else:
                    h_t[j], c_t[j] = self.LSTMLayers[j](h_t[j - 1], (h_t[j], c_t[j]))
            output = self.linear(h_t[-1])
            outputs.append(output)
        outputs = torch.stack(outputs, 1).squeeze(2) # convert to torch tensor
        return outputs








