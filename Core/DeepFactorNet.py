import torch
import torch.nn as nn
from Core.LSTMNet import LSTMNet
from Core.LSTMGaussianNet import LSTMGaussianNet

class DeepFactorNet(nn.Module):

    def __init__(self, numLayers=1, inputDim=1, hiddenDim=32, outputDim=1, device='cpu'):
        """An example LSTM network for deep factor model. Note that the network outputs both action and value
        # Argument
        numLstms: number of LSTM layers
        inputDim: dimensionality of input feature, for simplicity, the number of features are the number of time series
        hiddenDim: dimensionality of hidden feature in LSTM, which is also the number of global factors
        outputDim: dimensionality of output feature
        """
        super(DeepFactorNet, self).__init__()

        self.individualShocks = []
        # each individual shock will have a LSTM network
        for i in range(inputDim):
            self.individualShocks.append(LSTMGaussianNet(numLayers=numLayers, inputDim=inputDim, hiddenDim=hiddenDim, outputDim=1,
                                                    device=device, propogateMethod='sample', stdOnly = True))

        self.globalFactorNet = LSTMNet(numLayers=numLayers, inputDim=inputDim, hiddenDim=hiddenDim, outputDim=outputDim)

    def forward(self, X, future = 0):
        '''
        X: the time series input, size is given by batchSize, seqLen, stateDim
        future: number of step forward
        return mean, std
        mean is a deterministic process determined by the global factor dynamics
        std is a deterministic process determined by indivisual process
        '''
        means = self.globalFactorNet(X, future)
        stds = []
        for i in range(self.inputDim):
            stds.append(self.individualShocks[i](X[:,:,i]), future)


        outputs = torch.cat((means, stds), -1)
        return outputs

    def get_loss(self, means, stds, samples):

        diff = torch.sub(samples, means)
        loss = torch.mean(torch.div(torch.pow(diff, 2), torch.pow(stds, 2))) \
               + 2 * torch.mean(torch.log(stds))
        return loss
