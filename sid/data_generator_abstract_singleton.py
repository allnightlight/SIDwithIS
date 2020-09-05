'''
Created on 2020/09/04

@author: ukai
'''

import json

import numpy as np


class DataGeneratorAbstractSingleton(object):
    '''
    classdocs
    '''
    
    _instances = {}
    
    @classmethod
    def getInstance(cls, **params):
        key = json.dumps(params)
        
        if key in cls._instances:
            _inst = cls._instances[key]
        else:
            _inst = cls(**params)
            cls._instances[key] = _inst
        return _inst
        
        
    def __init__(self, Nsample=None, Ny=None, Nu=None):
        
        self.Nsample = Nsample
        self.Ny = Ny
        self.Nu = Nu
        self.alreadyLoaded = False
        
    def loadData(self):
        if self.alreadyLoaded:
            # skipped
            pass
        else:
            self.loadDataReal()
            self.alreadyLoaded = True
        
    def loadDataReal(self):
        
        print("""
Test data will be loaded with the following parameters:
    Nsample:   {0}
    Ny:        {1}
    Nu:        {2}
        """.format(self.Nsample, self.Ny, self.Nu))
        
        rstate = np.random.RandomState(seed=0)

        self.T = np.arange(self.Nsample) # (Nsample,)        
        self.Y = rstate.randn(self.Nsample, self.Ny) # (Nsample, Ny)
        self.U = rstate.randn(self.Nsample, self.Nu) # (Nsample, Nu)
        self.Ev = rstate.randint(2, size=(self.Nsample,)).astype(np.float) # (Nsample,)
        self.IsNaN = rstate.rand(self.Nsample) < 1/2**4 # (Nsample,)
        
        self.Y[self.IsNaN,:] = np.nan
        self.U[self.IsNaN,:] = np.nan
        self.Ev[self.IsNaN] = np.nan
        
        pass

        
