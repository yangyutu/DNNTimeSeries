import os
import torch
import json
import random
import numpy as np
from Models.RNNPredictor.RNNPredictorBase import RNNPredictorBase

class RNNPredictor(RNNPredictorBase):

    def __init__(self, config, net, optimizer, netLossFunc, trainDataLoader):
        super(RNNPredictor, self).__init__(config, net, optimizer, netLossFunc, trainDataLoader)

    def train_for_steps(self, nstep):

        for i in range(nstep):
            # see https://pytorch.org/docs/stable/optim.html for why we define a closure
            x, y = self.trainDataLoader.fetch_batch(self.trainBatchSize)
            x = torch.from_numpy(x).float().to(self.device)
            y = torch.from_numpy(y).float().to(self.device)

            #    optimizer.step(closure)
            self.optimizer.zero_grad()
            out = self.net(x)
            loss = self.netLossFunc(out, y)
            loss.backward()

            # clip gradient
            if self.netGradClip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.netGradClip)

            self.optimizer.step()
            self.learnStepCounter += 1
            if self.learnStepCounter % self.lossRecordStep == 0:
                if self.verbose:
                    print(self.learnStepCounter, loss.item())
                self.losses.append([self.learnStepCounter, loss.item()])

            if self.config['logFlag'] and self.learnStepCounter % self.config['logFrequency'] == 0:
                self.save_checkpoint()


    def predict(self, x, covariates = None, future = 0):
        if not covariates:
            return self.net.forward(x, future)
        else:
            return self.net.forward_with_covariates(x, covariates, future)






    def train(self):
        raise NotImplementedError


    def save_all(self, identifier=None):
        if identifier is None:
            identifier = self.identifier
        prefix = self.dirName + identifier + 'FinalStep' + str(self.learnStepCounter)
        torch.save({
            'learnStep': self.learnStepCounter,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, prefix + '_checkpoint.pt')

        self.saveLosses(prefix + '_loss.txt')

    def save_checkpoint(self, identifier=None):
        if identifier is None:
            identifier = self.identifier
        prefix = self.dirName + identifier + 'Step' + str(self.learnStepCounter)
        self.saveLosses(prefix + '_loss.txt')

        torch.save({
            'learnStep': self.learnStepCounter,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, prefix + '_checkpoint.pt')

    def load_checkpoint(self, prefix):


        checkpoint = torch.load(prefix + '_checkpoint.pt')
        self.learnStepCounter = checkpoint['learnStep']
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])