'''
Created on 2020/07/20

@author: ukai
'''


from sid_agent import SidAgent
from sid_batch_data_agent import SidBatchDataAgent
from sid_batch_data_environment import SidBatchDataEnvironment
import numpy as np
import torch
import torch.nn as nn


class SidAgent002(SidAgent, nn.Module):
    '''
    classdocs
    '''

    def __init__(self, Ny, Nu, Nhidden, dampingConstantInitial, use_offset_compensate):
        super(SidAgent002, self).__init__()

        self.Ny, self.Nu, self.Nhidden = Ny, Nu, Nhidden
        self.use_offset_compensate = use_offset_compensate

        self._lmbd_real_enc, self._lmbd_imag_enc, self._B_enc, self._C_enc, self._D_enc = self.initializeLtiParameters(Nhidden, Nu+Ny, Nhidden, dampingConstantInitial, use_D=False)
        self.param_enc = self._lmbd_real_enc, self._lmbd_imag_enc, self._B_enc, self._C_enc, self._D_enc
        
        self._lmbd_real_dec, self._lmbd_imag_dec, self._B_dec, self._C_dec, self._D_dec = self.initializeLtiParameters(Nhidden, Nu, Ny, dampingConstantInitial, use_D=True) 
        self.param_dec = self._lmbd_real_dec, self._lmbd_imag_dec, self._B_dec, self._C_dec, self._D_dec 
 
    # <<private>>
    def initializeLtiParameters(self, Nhidden, Nu, Ny, dampingConstantInitial, use_D):
        
        r = np.ones(Nhidden//2) * dampingConstantInitial # (Nhidden//2,)
        theta = np.random.rand(Nhidden//2) * np.pi/2 # (Nhidden//2,)
        
        lmbd_real = r * np.cos(theta) # (Nhidden//2)
        lmbd_imag = r * np.sin(theta) # (Nhidden//2)
        B = np.random.randn(Nhidden, Nu)/np.sqrt(Nu) * np.sqrt(1-dampingConstantInitial**2) # (Nhidden, Nu)

        _lmbd_real = nn.Parameter(torch.from_numpy(lmbd_real.astype(np.float32))) # (Nhidden//2,)
        _lmbd_imag = nn.Parameter(torch.from_numpy(lmbd_imag.astype(np.float32))) # (Nhidden//2,)
        _B = nn.Parameter(torch.from_numpy(B.astype(np.float32))) # (Nhidden, Nu)
        _C = nn.Linear(Nhidden, Ny)
        if use_D:
            _D = nn.Linear(Nu, Ny)
        else:
            _D = None
        
        return _lmbd_real, _lmbd_imag, _B, _C, _D

    # <<private>>
    def createAmatrix(self, _lmbd_real, _lmbd_imag):
        _A11 = torch.diag(_lmbd_real) # = A22, (Nhidden//2, Nhidden//2)
        _A21 = torch.diag(_lmbd_imag) # = A21, (Nhidden//2, Nhidden//2)
        _A = torch.cat((torch.cat((_A11, -_A21), dim=1), 
            torch.cat((_A21, _A11), dim=1)), dim=0) #(Nhidden, Nhidden)
        return _A

    # <<private>>
    def seq2seq(self, _A, _B, _U, _xinit):
        
        N = _U.shape[0]
        
        _x = _xinit # (*, Nhidden)
        _Bu = torch.matmul(_U, _B.t()) # (N0, *, Nhidden)        
        
        X = [_x,]
        for k1 in range(N):          
            _x = torch.matmul(_x, _A.t()) + _Bu[k1,...] # (*, Nhidden)
            X.append(_x)      
        _X = torch.stack(X, dim=0) # (N+1, *, Nhidden)
        return _X # (N+1, *, Nhidden)

    def forward(self, batchDataIn):        
        assert isinstance(batchDataIn, SidBatchDataEnvironment)
                
        if self.use_offset_compensate:
            _Y0org = batchDataIn._Y0 # (N0, *, Ny)
            _U0org = batchDataIn._U0 # (N0, *, Nu)        
            _U1org = batchDataIn._U1 # (N1, *, Nu)
            
            _yOffset = torch.mean(_Y0org, dim=0) # (*, Ny)
            _uOffset = torch.mean(_U0org, dim=0) # (*, Nu)
            
            _Y0 = _Y0org - _yOffset # (N0, *, Ny)
            _U0 = _U0org - _uOffset # (N0, *, Nu)
            _U1 = _U1org - _uOffset # (N0, *, Nu)            
        else:
            _Y0 = batchDataIn._Y0 # (N0, *, Ny)
            _U0 = batchDataIn._U0 # (N0, *, Nu)        
            _U1 = batchDataIn._U1 # (N1, *, Nu)

        Nbatch = _U0.shape[1]        
                
        # Encoding        
        _lmbd_real, _lmbd_imag, _B, _C, _D = self.param_enc        
        _A = self.createAmatrix(_lmbd_real, _lmbd_imag) # (Nhidden, Nhidden)        
        _X0 = self.seq2seq(_A, _B, torch.cat((_U0, _Y0), axis=-1), torch.zeros((Nbatch, self.Nhidden))) # (N0+1, *, Nhidden)
        _x1init = _C(_X0[-1,...]) # (*, Nhidden)
                
        # Decoding
        _lmbd_real, _lmbd_imag, _B, _C, _D = self.param_dec
        _A = self.createAmatrix(_lmbd_real, _lmbd_imag) # (Nhidden, Nhidden)        
        _X2 = self.seq2seq(_A, _B, _U1, _x1init) # (N1+1, *, Nhidden)
        _U2 = torch.cat((_U0[-1,None,:], _U1)) # (N2 = N1 + 1, * , Nu)
        _Yhat2 = _C(_X2) + _D(_U2) # (N2, *, Ny)
                
        if self.use_offset_compensate:
            _Yhat2 = _Yhat2 + _yOffset # (N2, *, Ny)
        
        batchDataOut = SidBatchDataAgent(_Yhat2, T = batchDataIn.T2)
        
        return batchDataOut