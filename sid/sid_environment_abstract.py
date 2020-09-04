'''
Created on 2020/09/01

@author: ukai
'''
from builtins import isinstance

import torch

from data_generator_abstract_singleton import DataGeneratorAbstractSingleton
import numpy as np
from sid_batch_data_environment import SidBatchDataEnvironment
from sl_environment import SlEnvironment


class SidEnvironmentAbstract(SlEnvironment):
    '''
    classdocs
    '''

    def __init__(self, dataGeneratorSingleton, Ntrain, Nbatch, N0, N1):
        SlEnvironment.__init__(self)
        
        assert isinstance(dataGeneratorSingleton, DataGeneratorAbstractSingleton)
        
        self.Ny = dataGeneratorSingleton.Ny
        self.Nu = dataGeneratorSingleton.Nu
        
        self.dataGeneratorSingleton = dataGeneratorSingleton
        assert Ntrain < dataGeneratorSingleton.Nsample
        assert Ntrain > N0 + N1
        self.Nsample = dataGeneratorSingleton.Nsample
        self.Ntrain = Ntrain
        self.Ntest = dataGeneratorSingleton.Nsample-Ntrain
        assert self.Ntest > N0 + N1
        self.Nbatch = Nbatch
        self.N0 = N0 # estimation horizon's length
        self.N1 = N1 # prediction horizon's length
        
    # <<protected>>
    def getAvailableIndex(self, segment):
        
        idx = None
        if segment == "train":
            idx =np.arange(self.N0, self.Ntrain-self.N1) # (* = Ntrain-N0-N1)
        
        if segment == "test":
            idx =np.arange(self.Ntrain+self.N0, self.Nsample-self.N0-self.N1) # (* = Ntest-N0-N1)
        
        idx = idx.reshape((1,-1)) + np.arange(-self.N0, self.N1).reshape((-1, 1)) # (N0+N1, *)
        idxAvailable = idx[:, ~np.any(self.dataGeneratorSingleton.IsNaN[idx], axis=0)] # (N0+N1, Navailable), * = Navailable + #NaN
        
        return idxAvailable # (N0+N1, Navailable)
        
    # <<protected>>
    def extractBatchData(self, idx):
        # idx: (N0+N1, *)
        
        N0 = self.N0
        
        U0batch = self.dataGeneratorSingleton.U[idx[:N0,:],:] # (N0, *, Nu)
        U1batch = self.dataGeneratorSingleton.U[idx[N0:,:],:] # (N1, *, Nu)
        
        Y0batch = self.dataGeneratorSingleton.Y[idx[:N0,:],:] # (N0, *, Ny)
        Y2batch = self.dataGeneratorSingleton.Y[idx[(N0-1):,:],:] # (N2, *, Ny)
        
        Ev0batch = self.dataGeneratorSingleton.Ev[idx[:N0,:]] # (N0, *)
        Ev1batch = self.dataGeneratorSingleton.Ev[idx[N0:,:]] # (N1, *)
        
        _U0 = torch.tensor(U0batch.astype(np.float32))
        _U1 = torch.tensor(U1batch.astype(np.float32))
        _Ev0 = torch.tensor(Ev0batch.astype(np.float32))
        _Ev1 = torch.tensor(Ev1batch.astype(np.float32))
        _Y0 = torch.tensor(Y0batch.astype(np.float32))
        _Y2 = torch.tensor(Y2batch.astype(np.float32))
        
        batchDataEnvironment = SidBatchDataEnvironment( _U0, _Ev0, _Y0, _U1, _Ev1, _Y2)
        
        return batchDataEnvironment

    def generateBatchDataIterator(self):
        
        raise NotImplementedError()
    
    def getTestBatchData(self):
        
        idxAvailable = self.getAvailableIndex("test") # (N0+N1, Navailable) 
        return self.extractBatchData(idxAvailable)