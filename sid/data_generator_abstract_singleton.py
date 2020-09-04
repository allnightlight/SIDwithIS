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
        