'''
Created on 2020/09/04

@author: ukai
'''
import numpy as np
from sid_environment_abstract import SidEnvironmentAbstract

class SidEnvironmentNormalSampling(SidEnvironmentAbstract):
    '''
    classdocs
    '''


    def __init__(self, dataGeneratorSingleton, Ntrain, Nbatch, N0, N1):
        SidEnvironmentAbstract.__init__(self, dataGeneratorSingleton, Ntrain, Nbatch, N0, N1)
        
        
    def generateBatchDataIterator(self):

        Nbatch = self.Nbatch
        idxAvailable = self.getAvailableIndex("train") # (N0+N1, Navailable) 
        Navailable = idxAvailable.shape[1]
        
        for _ in range(Navailable//Nbatch):
            idx = idxAvailable[:, np.random.randint(low=0, high=Navailable, size=(Nbatch,))] # (N0+N1, *)
            yield self.extractBatchData(idx)
