'''
Created on 2020/07/16

@author: ukai
'''

from builtins import isinstance

import torch
from torch.optim import Adam

from sid_agent import SidAgent
from sid_batch_data_agent import SidBatchDataAgent
from sid_batch_data_environment import SidBatchDataEnvironment
from sid_environment import SidEnvironment
from sl_trainer import SlTrainer


class SidTrainer(SlTrainer):
    '''
    classdocs
    '''


    def __init__(self, agent, environment):
        SlTrainer.__init__(self, agent, environment)
        
        assert isinstance(agent, SidAgent)
        assert isinstance(environment, SidEnvironment)
        
        self.optimizer = Adam(agent.parameters())
        
        
    def update(self, batchDataIn, batchDataOut):
        
        assert isinstance(batchDataIn, SidBatchDataEnvironment)
        assert isinstance(batchDataOut, SidBatchDataAgent)
        
        _Y = batchDataIn._Y2 # (Nhrz+1, *, Ny)
        _Yhat = batchDataOut._Yhat # (Nhrz+1, * , Ny)
        
        _loss = torch.mean((_Y - _Yhat)**2)
        
        self.optimizer.zero_grad()
        _loss.backward()
        self.optimizer.step() 