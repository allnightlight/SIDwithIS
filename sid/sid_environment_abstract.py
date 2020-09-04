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

    def __init__(self, dataGeneratorSingleton, Ntrain, Nbatch, N0, N1, sampling_balance):
        SlEnvironment.__init__(self)
        
        assert isinstance(dataGeneratorSingleton, DataGeneratorAbstractSingleton)
        
        self.Ny = dataGeneratorSingleton.Ny
        self.Nu = dataGeneratorSingleton.Nu
        
        self.dataGeneratorSingleton = dataGeneratorSingleton
        self.Ntrain = Ntrain
        self.Nbatch = Nbatch
        self.N0 = N0 # estimation horizon's length
        self.N1 = N1 # prediction horizon's length
        self.sampling_balance = sampling_balance # the proportional rate of samples with ev == 1 in a batch data
        
    def generateBatchDataIterator(self):


        Nbatch = self.Nbatch 
        N0 = self.N0
        N1 = self.N1 
        
        for _ in range((self.Ntrain-N0-N1)//Nbatch):
            idx = np.random.randint(low=0, high=self.Ntrain-N0-N1, size=(Nbatch,))
            idx = idx.reshape((1,-1)) + np.arange(N0+N1).reshape(-1,1) # (N0+N1, Nbatch)
            U0batch = self.dataGeneratorSingleton.U[idx[:N0,:],:] # (N0, *, Nu)
            U1batch = self.dataGeneratorSingleton.U[idx[N0:,:],:] # (N1, *, Nu)
            
            Y0batch = self.dataGeneratorSingleton.Y[idx[:N0,:],:] # (N0, *, Ny)
            Y2batch = self.dataGeneratorSingleton.Y[idx[(N0-1):,:],:] # (N2, *, Ny)
            
            Ev0batch = self.dataGeneratorSingleton.Ev[idx[:N0,:]] # (N0, *)
            Ev1batch = self.dataGeneratorSingleton.Ev[idx[N0:,:]] # (N1, *)
            
            _U0 = torch.tensor(U0batch)
            _U1 = torch.tensor(U1batch)
            _Ev0 = torch.tensor(Ev0batch)
            _Ev1 = torch.tensor(Ev1batch)
            _Y0 = torch.tensor(Y0batch)
            _Y2 = torch.tensor(Y2batch)
            
            batchDataEnvironment = SidBatchDataEnvironment( _U0, _Ev0, _Y0, _U1, _Ev1, _Y2)
            
            yield batchDataEnvironment
    
    def getTestBatchData(self):
        
        N0 = self.N0
        N1 = self.N1 
        
        idx = np.arange(self.Ntrain+N0, self.Ntrain+self.Ntest-N1) # (*, ), where Nbatch = Ntest - N0 - N1.
        idx = idx.reshape((1,-1)) + np.arange(N0+N1).reshape(-1,1) # (N0+N1, *)
        U0batch = self.dataGeneratorSingleton.U[idx[:N0,:],:] # (N0, *, Nu)
        U1batch = self.dataGeneratorSingleton.U[idx[N0:,:],:] # (N1, *, Nu)
        
        Y0batch = self.dataGeneratorSingleton.Y[idx[:N0,:],:] # (N0, *, Ny)
        Y2batch = self.dataGeneratorSingleton.Y[idx[(N0-1):,:],:] # (N2, *, Ny)

        Ev0batch = self.dataGeneratorSingleton.Ev[idx[:N0,:]] # (N0, *)
        Ev1batch = self.dataGeneratorSingleton.Ev[idx[N0:,:]] # (N1, *)

        _U0 = torch.tensor(U0batch)
        _U1 = torch.tensor(U1batch)
        _Ev0 = torch.tensor(Ev0batch)
        _Ev1 = torch.tensor(Ev1batch)
        _Y0 = torch.tensor(Y0batch)
        _Y2 = torch.tensor(Y2batch)
                
        batchDataEnvironment = SidBatchDataEnvironment( _U0, _Ev0, _Y0, _U1, _Ev1, _Y2)
        
        return batchDataEnvironment