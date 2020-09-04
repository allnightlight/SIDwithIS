'''
Created on 2020/09/04

@author: ukai
'''

import numpy as np
from sid_environment_abstract import SidEnvironmentAbstract

class SidEnvironmentImbalancedSampling(SidEnvironmentAbstract):
    '''
    classdocs
    '''


    def __init__(self, dataGeneratorSingleton, Ntrain, Nbatch, N0, N1, sampling_balance):
        SidEnvironmentAbstract.__init__(self, dataGeneratorSingleton, Ntrain, Nbatch, N0, N1)
        self.sampling_balance = sampling_balance # the proportional rate of samples with ev == 1 in a batch data
        
        
    def generateBatchDataIterator(self):

        N0 = self.N0
        Nbatch = self.Nbatch
        idxAvailable = self.getAvailableIndex("train") # (N0+N1, Navailable) 
        
        idxAvailableEvOn = idxAvailable[:, self.dataGeneratorSingleton.Ev[idxAvailable[N0, :]] == 1] # (N0+N1, NavailableEvOn)
        idxAvailableEvOff = idxAvailable[:, self.dataGeneratorSingleton.Ev[idxAvailable[N0, :]] == 0] # (N0+N1, NavailableEvOff)
        
        NbatchEvOn = int(self.sampling_balance * Nbatch)
        NbatchEvOff = Nbatch - NbatchEvOn
        
        NavailableEvOn = idxAvailableEvOn.shape[1]
        
        for _ in range(NavailableEvOn//Nbatch):
            
            idxEvOn = idxAvailableEvOn[:, np.random.randint(low=0, high=idxAvailableEvOn.shape[1], size=(NbatchEvOn,))] # (N0+N1, NbatchEvOn)
            idxEvOff = idxAvailableEvOff[:, np.random.randint(low=0, high=idxAvailableEvOff.shape[1], size=(NbatchEvOff,))] # (N0+N1, NbatchEvOff)
            idx = np.concatenate((idxEvOn, idxEvOff), axis=1) # (N0+N1, Nbatch)
            
            yield self.extractBatchData(idx)
        