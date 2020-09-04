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
        
        
    def __init__(self, Nsample, Ny, Nu):
        
        self.Nsample = Nsample
        self.Ny = Ny
        self.Nu = Nu
        
        rstate = np.random.RandomState(seed=0)
        
        self.Y = rstate.randn(Nsample, Ny) # (Nsample, Ny)
        self.U = rstate.randn(Nsample, Nu) # (Nsample, Nu)
        self.Ev = rstate.randint(2, size=(Nsample,)) # (Nsample,)
        self.IsNaN = rstate.rand(Nsample) < 1/2**4 # (Nsample,)
        

        
