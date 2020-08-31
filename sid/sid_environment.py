'''
Created on 2020/07/16

@author: ukai
'''
import torch

from data_generator_singleton import DataGeneratorSingleton
import numpy as np
from sid_batch_data_environment import SidBatchDataEnvironment
from sl_environment import SlEnvironment


class SidEnvironment(SlEnvironment):
    '''
    classdocs
    '''


    def __init__(self, Nhidden, Ntrain, Ntest, T0, T1, Ny, Nu, Nbatch, N0, N1, seed):
        '''
        Constructor
        '''
        
        super(SidEnvironment, self).__init__()
        
        self.dataGeneratorSingleton = DataGeneratorSingleton.getInstance(
                                            Nhidden=Nhidden
                                           , Ntrain=Ntrain + Ntest
                                           , T0=T0
                                           , T1=T1
                                           , Ny=Ny
                                           , Nu=Nu
                                           , seed=seed)
        
        self.Ntrain = Ntrain
        self.Ntest = Ntest
        self.Nbatch = Nbatch
        self.N0 = N0
        self.N1 = N1
        
        
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
            
            _U0 = torch.tensor(U0batch)
            _U1 = torch.tensor(U1batch)            
            _Y0 = torch.tensor(Y0batch)
            _Y2 = torch.tensor(Y2batch)
            
            batchDataEnvironment = SidBatchDataEnvironment( _U0, _Y0, _U1, _Y2)
            
            yield batchDataEnvironment
    
    def getTestBatchData(self):
        
        N0 = self.N0
        N1 = self.N1 
        
        idx = np.arange(self.Ntrain, self.Ntrain+self.Ntest-N0-N1) # (*, ), where Nbatch = Ntest - N0 - N1.
        idx = idx.reshape((1,-1)) + np.arange(N0+N1).reshape(-1,1) # (N0+N1, *)
        U0batch = self.dataGeneratorSingleton.U[idx[:N0,:],:] # (N0, *, Nu)
        U1batch = self.dataGeneratorSingleton.U[idx[N0:,:],:] # (N1, *, Nu)
        
        Y0batch = self.dataGeneratorSingleton.Y[idx[:N0,:],:] # (N0, *, Ny)
        Y2batch = self.dataGeneratorSingleton.Y[idx[(N0-1):,:],:] # (N2, *, Ny)
        
        _U0 = torch.tensor(U0batch)
        _U1 = torch.tensor(U1batch)            
        _Y0 = torch.tensor(Y0batch)
        _Y2 = torch.tensor(Y2batch)
        
        batchDataEnvironment = SidBatchDataEnvironment( _U0, _Y0, _U1, _Y2)
        
        return batchDataEnvironment