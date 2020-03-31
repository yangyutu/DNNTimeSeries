import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, numLayers=1, inputDim=1, hiddenDim=32, outputDim=1, device='cpu'):
        """An example MLP network for actor-critic learning. Note that the network outputs both action and value
        # Argument
        numLstms: number of LSTM layers
        inputDim: dimensionality of input feature
        hiddenDim: dimensionality of hidden feature in LSTM
        outputDim: dimensionality of output feature
        """
        super(RNN, self).__init__()
        self.device = device
        self.numLayers = numLayers
        self.hiddenDim = hiddenDim
        self.RNNLayers = torch.nn.ModuleList()
        for i in range(numLayers):
            if i == 0:
                RNNLayer = nn.RNNCell(inputDim, hiddenDim)
            else:
                RNNLayer = nn.RNNCell(hiddenDim, hiddenDim)

            self.RNNLayers.append(RNNLayer)

        self.linear = nn.Linear(hiddenDim, outputDim)


    def forward(self, x, future = 0):

        outputs = []
        h_t = []
        for _ in range(self.numLayers):
            h_t.append(torch.zeros(x.size(0), self.hiddenDim, dtype=torch.float, device = self.device))

        for i, x_t in enumerate(x.chunk(x.size(1), dim=1)):
            for j in range(self.numLayers):
                if j == 0:
                    h_t[j] = self.RNNLayers[j](x_t, h_t[j])
                else:
                    h_t[j] = self.RNNLayers[j](h_t[j - 1], h_t[j])
            output = self.linear(h_t[-1])
            outputs.append(output)

        for i in range(future):
            for j in range(self.numLayers):
                if j == 0:
                    h_t[j] = self.RNNLayers[j](output, h_t[j]) #  use final output as input
                else:
                    h_t[j] = self.RNNLayers[j](h_t[j - 1], h_t[j])
            output = self.linear(h_t[-1])
            outputs.append(output)
        outputs = torch.stack(outputs, 1).squeeze(2) # convert to torch tensor
        return outputs








