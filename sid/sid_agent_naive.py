'''
Created on 2020/09/07

@author: ukai
'''

import torch

import numpy as np
from sid_agent import SidAgent
from sid_batch_data_agent import SidBatchDataAgent
from sid_batch_data_environment import SidBatchDataEnvironment
import torch.nn as nn


class SidAgentNaive(SidAgent, nn.Module):
    '''
    classdocs
    '''

    def __init__(self):
        super(SidAgentNaive, self).__init__()
        
        # This not used parameter is defined in order to pass the definition of trainer
        self._dummy = nn.Parameter(torch.rand(3))  
        
    def forward(self, batchDataIn):        
        assert isinstance(batchDataIn, SidBatchDataEnvironment)
                
        N1 = batchDataIn._U0.shape[0]   
        _Yhat2 = batchDataIn._Y0[-1,:,:].repeat(N1,1,1) # (N1, *, Ny)                
        batchDataOut = SidBatchDataAgent(_Yhat2, T = batchDataIn.T2)

        return batchDataOut
