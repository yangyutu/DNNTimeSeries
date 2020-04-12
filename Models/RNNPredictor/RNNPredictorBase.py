import os
import torch
import json
import random
import numpy as np
import math

class RNNPredictorBase(object):
    """Abstract base class for RNN predictor based.
        This base class contains common routines to perform basic initialization, action selection, and logging
        # Arguments
            config: a dictionary for training parameters
            policyNet: neural network for Q learning
            targetNet: a slowly changing policyNet to provide Q value targets
            env: environment for the agent to interact. env should implement same interface of a gym env
            optimizer: a network optimizer
            netLossFunc: loss function of the network, e.g., mse
            nbAction: number of actions
            stateProcessor: a function to process output from env, processed state will be used as input to the networks
            experienceProcessor: additional steps to process an experience
        """
    def __init__(self, config, net, optimizer, netLossFunc, trainDataLoader):
        self.config = config
        self.read_config()
        self.net = net
        self.optimizer = optimizer
        self.netLossFunc = netLossFunc
        self.trainDataLoader = trainDataLoader
        self.initialization()

    def initialization(self):
        # move model to correct device
        self.net = self.net.to(self.device)

        self.dirName = 'Log/'
        if 'dataLogFolder' in self.config:
            self.dirName = self.config['dataLogFolder']
        if not os.path.exists(self.dirName):
            os.makedirs(self.dirName)

        self.identifier = ''
        self.learnStepCounter = 0  #for target net update
        self.losses = []

    def read_config(self):
        '''
        read parameters from self.config object
        initialize various flags and parameters
        trainStep: number of episodes to train
        targetNetUpdateStep: frequency in terms of training steps/episodes to reset target net
        trainBatchSize: mini batch size for gradient decent.
        gamma: discount factor
        netGradClip: gradient clipping parameter
        netUpdateOption: allowed strings are targetNet, policyNet, doubleQ
        verbose: bool, default false.
        nStepForward: multiple-step forward Q learning, default 1
        lossRecordStep: frequency to store loss.
        episodeLength: maximum steps in an episode
        netUpdateFrequency: frequency to perform gradient decent
        netUpdateStep: number of steps for gradient decent
        epsThreshold: const epsilon used throughout the training. Will be overridden by epsilon start, epsilon end, epsilon decay
        epsilon_start: start epsilon for scheduled epsilon exponential decay
        epsilon_final: end epsilon for scheduled epsilon exponential decay
        epsilon_decay: factor for exponential decay of epsilon
        device: cpu or cuda
        randomSeed
        hindSightER: bool variable for hindsight experience replay
        hindSightERFreq: frequency to perform hindsight experience replay
        return: None
        '''
        self.trainStep = self.config['trainStep']

        self.trainBatchSize = self.config['trainBatchSize']
        self.netGradClip = None
        if 'netGradClip' in self.config:
            self.netGradClip = self.config['netGradClip']
        self.lossRecordStep = 500
        if 'lossRecordStep' in self.config:
            self.lossRecordStep = self.config['lossRecordStep']

        self.device = 'cpu'
        if 'device' in self.config and torch.cuda.is_available():
            self.device = self.config['device']


        self.randomSeed = 1
        if 'randomSeed' in self.config:
            self.randomSeed = self.config['randomSeed']

        self.verbose = False
        if 'verbose' in self.config:
            self.verbose = self.config['verbose']

        random.seed(self.randomSeed)

    def train_for_steps(self, nstep):
        raise NotImplementedError


    def train(self):
        raise NotImplementedError



    def save_all(self):
        raise NotImplementedError

    def save_checkpoint(self):
        raise NotImplementedError

    def load_checkpoint(self, prefix):
        raise NotImplementedError

    def save_config(self, fileName):
        with open(fileName, 'w') as f:
            json.dump(self.config, f)

    def saveLosses(self, fileName):
        np.savetxt(fileName, np.array(self.losses), fmt='%.5f', delimiter='\t')

    def loadLosses(self, fileName):
        self.losses = np.genfromtxt(fileName).tolist()
