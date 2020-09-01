'''
Created on 2020/09/01

@author: ukai
'''
import torch

from data_generator_singleton import DataGeneratorSingleton
import numpy as np
from sid_batch_data_environment import SidBatchDataEnvironment
from sid_environment_abstract import SidEnvironmentAbstract


class SidEnvironmentImbalancedSampling(SidEnvironmentAbstract):
    '''
    classdocs
    '''


    def __init__(self, Nhidden, Ntrain, Ntest, T0, T1, Ny, Nu, Nw, prob_step, Nbatch, N0, N1, sampling_balance, seed):
        '''
        Constructor
        '''
        
        super(SidEnvironmentImbalancedSampling, self).__init__()
        
        self.dataGeneratorSingleton = DataGeneratorSingleton.getInstance(
                                            Nhidden=Nhidden
                                           , Ntrain=Ntrain + Ntest
                                           , T0=T0
                                           , T1=T1
                                           , Ny=Ny
                                           , Nu=Nu
                                           , Nw=Nw
                                           , prob_step=prob_step
                                           , action_distribution="step"
                                           , seed=seed)
        
        self.Ntrain = Ntrain
        self.Ntest = Ntest
        self.Nbatch = Nbatch
        self.N0 = N0
        self.N1 = N1
        self.sampling_balance = sampling_balance
        
        
    def generateBatchDataIterator(self):


        Nbatch = self.Nbatch
        NbatchEvOn = int(self.Nbatch * self.sampling_balance) # the proportion of sampling in a batch with ev[t] = 1
        NbatchEvOff = Nbatch - NbatchEvOn
        
        N0 = self.N0
        N1 = self.N1 
        
        idxEvOn = np.where(self.dataGeneratorSingleton.Ev == 1)[0]
        idxEvOff = np.where(self.dataGeneratorSingleton.Ev == 0)[0]
        idxEvOff = idxEvOff[(idxEvOff >= N0) & (idxEvOff <= self.Ntrain-N1)] # [N0, Ntrain - N1]
        idxEvOn = idxEvOn[(idxEvOn >= N0) & (idxEvOn <= self.Ntrain-N1)] # [N0, Ntrain - N1]
        
        for _ in range((self.Ntrain-N0-N1)//Nbatch):
            idx = np.concatenate((np.random.choice(idxEvOn, size=(NbatchEvOn,)), np.random.choice(idxEvOff, size=(NbatchEvOff,))))         
            idx = idx.reshape((1,-1)) + np.arange(-N0, N1).reshape(-1,1) # (N0+N1, Nbatch), [0, Ntrain-1]
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
        
        idxEvOn = np.where(self.dataGeneratorSingleton.Ev == 1)[0]
        idxEvOff = np.where(self.dataGeneratorSingleton.Ev == 0)[0]
        idxEvOff = idxEvOff[(idxEvOff >= self.Ntrain + N0) & (idxEvOff <= self.Ntest + self.Ntrain-N1)] # [Ntrain+N0, Ntest+Ntrain-N1]
        idxEvOn = idxEvOn[(idxEvOn >= self.Ntrain + N0) & (idxEvOn <= self.Ntest + self.Ntrain-N1)] # [Ntrain+N0, Ntest+Ntrain-N1]
 
        idx = idxEvOn
        idx = idx.reshape((1,-1)) + np.arange(-N0, N1).reshape(-1,1) # (N0+N1, *) [Ntrain, Ntest+Ntrain-1]
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