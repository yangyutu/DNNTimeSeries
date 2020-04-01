import numpy as np
import random
from matplotlib import pyplot as plt

class SineWaveData:
    """Loads the modulaed sinewave dataset. See _time_series function below for the functional form"""

    def __init__(self, numSamples, conditionLength, predictionLen, covariateFlag = False, seed = 1, noiseFlag = False, noiseStrength = 0.1):
        """
        Samples a modulated sine function and stores the samples in the dataset array. Batches are drawn out
        of this array during training.
        :param dev: device (CPU/GPU). Output of torch.device
        :param ctx_win_len: length of the context window (conditioning + prediction window)
        :param num_time_indx: number of time indices. Usually a relative time index, but can also include absolute time
        index from the beginning of the series (age). If the age is included, num_time_indx = 2, otherwise 1
        :param t_min: start time index of the modulated SW
        :param t_max: end time index of the modulated SW
        :param resolution: resolution of the modulated SW
        """

        self.stateDim = 1 if not covariateFlag else 2
        self.outputDim = 1
        self.T = 20 # period
        self.numSamples = numSamples
        self.conditionLen = conditionLength
        self.preditionLeng = predictionLen # for covariates
        np.random.seed(seed)
        x = np.empty((self.numSamples, self.conditionLen), 'int64')

        x[:] = np.array(range(self.conditionLen)) + np.random.randint(-4 * self.T, 4 * self.T, self.numSamples).reshape(self.numSamples, 1)
        data = np.sin(x / 1.0 / self.T).astype('float32')
        if noiseFlag:
            data = data + np.random.normal(0, noiseStrength, data.shape)

        # input and target are offset by 1
        self.x = data[:, :-1]
        self.y = data[:, 1:]


    def fetch_batch(self, batchSize):
        idx = random.sample(range(self.numSamples), k = batchSize)
        return self.x[idx], self.y[idx]

    def fetch_all(self):
        return self.x, self.y







