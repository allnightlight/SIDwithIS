'''
Created on 2020/07/16

@author: ukai
'''

import torch

import numpy as np
from sid_agent import SidAgent
from sid_batch_data_agent import SidBatchDataAgent
from sid_batch_data_environment import SidBatchDataEnvironment
import torch.nn as nn


class SidAgent001(SidAgent, nn.Module):
    '''
    classdocs
    '''

    def __init__(self, Ny, Nu, Nhidden, use_offset_compensate):
        super(SidAgent001, self).__init__()
        
        self.Ny, self.Nu, self.Nhidden = Ny, Nu, Nhidden
         
        self.xyu2x_enc = nn.Linear(Nhidden+Ny+Nu, Nhidden)
        self.x2x_enc = nn.Linear(Nhidden, Nhidden)
        
        self.xu2x_dec = nn.Linear(Nhidden+Nu, Nhidden)
        self.xu2y_dec = nn.Linear(Nhidden + Nu, Ny)
        
        self.Nhidden = Nhidden
        self.use_offset_compensate = use_offset_compensate # (True or False)


    def forward(self, batchDataIn):        
        assert isinstance(batchDataIn, SidBatchDataEnvironment)
        
        
        if self.use_offset_compensate:
            _Y0org = batchDataIn._Y0 # (N0, *, Ny)
            _U0org = batchDataIn._U0 # (N0, *, Nu)        
            _U1org = batchDataIn._U1 # (N1, *, Nu)
            
            _yOffset = _Y0org[-1,:,:] # (*, Ny)
            _uOffset = _U0org[-1,:,:] # (*, Nu)
            
            _Y0 = _Y0org - _yOffset # (N0, *, Ny)
            _U0 = _U0org - _uOffset # (N0, *, Nu)
            _U1 = _U1org - _uOffset # (N0, *, Nu)            
        else:
            _Y0 = batchDataIn._Y0 # (N0, *, Ny)
            _U0 = batchDataIn._U0 # (N0, *, Nu)        
            _U1 = batchDataIn._U1 # (N1, *, Nu)
        
        N0 = _U0.shape[0]
        N1 = _U1.shape[0]
        Nbatch = _U0.shape[1]
        
        _x = torch.zeros((Nbatch, self.Nhidden)) # (*, Nhidden)
        for k1 in range(N0):
            _u = _U0[k1,:] # (*, Nu)
            _y = _Y0[k1,:] # (*, Ny)
            _x = self.xyu2x_enc(torch.cat((_x, _y, _u), dim=1)) # (*, Nhidden)
        _x0hat = self.x2x_enc(_x) # (*, Nhidden)
 
        _x = _x0hat # (*, Nhidden)
        X2 = []
        X2.append(_x)
        for k1 in range(N1):
            _u = _U1[k1,:] # (*, Nu)
            _x = self.xu2x_dec(torch.cat((_x, _u), dim=1)) # (*, Nhidden)
            X2.append(_x)
        _U2 = torch.cat((_U0[-1,None,:], _U1)) # (N2 = N1 + 1, * , Nu)
        _X2 = torch.stack(X2, dim=0) # (N2 = N1 + 1, *, Nhidden)
        _Yhat2 = self.xu2y_dec(torch.cat((_X2, _U2), axis=-1)) # (N2, *, Ny)
        
        if self.use_offset_compensate:
            _Yhat2 = _Yhat2 + _yOffset # (N2, *, Ny)
        
        batchDataOut = SidBatchDataAgent(_Yhat2)

        return batchDataOut
