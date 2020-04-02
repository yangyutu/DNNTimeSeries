import torch
import torch.nn as nn
import torch.nn.functional as F



class LSTNet(nn.Module):

    def __init__(self, inputDim=1, CNNOutputChannel = 32, CNNKernelSize = 2, RNNHiddenDim=32, outputDim=1, skip = 1, AROrder=1, device='cpu', outputFunc = 'linear'):
        super(LSTNet, self).__init__()

        self.device = device

        self.inputDim = inputDim
        self.outputDim = outputDim

        self.hiddenSkip = 5
        self.AROrder = AROrder
        self.skip = skip
        self.CNNKernelSize = CNNKernelSize # how many time step will be convolve
        self.CNNOutputChannel = CNNOutputChannel
        self.RNNHiddenDim = RNNHiddenDim
        self.conv1 = nn.Conv2d(1, self.CNNOutputChannel, kernel_size=(self.CNNKernelSize, self.inputDim))
        self.GRU1 = nn.GRU(self.CNNOutputChannel, self.RNNHiddenDim, batch_first=True)

        if skip > 0:
            self.GRUSkip = nn.GRU(self.CNNOutputChannel, self.hiddenSkip)
            self.linear1 = nn.Linear(self.RNNHiddenDim + self.hiddenSkip * self.skip)
        else:
            self.linear1 = nn.Linear(self.RNNHiddenDim, self.outputDim)

        self.output = None
        if (outputFunc == 'sigmoid'):
            self.output = F.sigmoid
        if (outputFunc == 'tanh'):
            self.output = F.tanh

        # here we assume different features share the same AR coefficient
        if (self.AROrder > 0):
            self.ARLayer = nn.Linear(self.AROrder, 1);

    def forward(self, x):
        # x has the size of batchSize, SeqLen, StateDim

        batch_size = x.size(0)
        seqLen = x.size(1)

        # CNN
        # convert to the shape for CNN
        # c will have size batchSize, channel = 1, seqLen, stateDim
        c = x.view(-1, 1, seqLen, self.inputDim)
        # after convolution, c will have size batchSize, outputChannel, reduced seqLen, 1
        c = F.relu(self.conv1(c))
        #c = self.dropout(c)
        c = torch.squeeze(c, 3)
        #because kenelSize has one size of inputDim, third Dim will become 1

        # RNN
        # r has the batchSize, seqLen, channel (number of channels are the number of feature)
        r = c.permute(0, 2, 1).contiguous()
        _, r = self.GRU1(r)
        # r has the batchSize, 1, hiddenDim, so we need to squeeze
        r = r.squeeze(1)

        # skip-rnn
        pt = c.size(2) / self.skip
        if (self.skip > 0):
            s = c[:, :, int(-pt * self.skip):].contiguous()
            s = s.view(batch_size, self.CNNOutputChannel, pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.CNNOutputChannel)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.RNNHiddenDim)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)
        # c has batchSize, channel, and seqLen

        res = self.linear1(r)

        #AR portion
        if (self.AROrder > 0):
            # take the last AROrder time series
            z = x[:, -self.AROrder:, :]
            z = z.permute(0,2,1).contiguous().view(-1, self.AROrder)
            z = self.ARLayer(z)
            z = z.view(-1,self.outputDim)
            res = res + z




        if self.output:
            res = self.output(res)


        return res




