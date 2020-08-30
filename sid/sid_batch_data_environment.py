'''
Created on 2020/07/16

@author: ukai
'''
from batch_data_environment import BatchDataEnvironment

class SidBatchDataEnvironment(BatchDataEnvironment):
    '''
    classdocs
    '''


    def __init__(self, _U0, _Y0, _U1, _Y2):
        # _U0: (N0, *, Nu), _Y0: (N0, *, Ny)
        # _U1: (N1, *, Nu), _Y2: (N2, *, Ny), N2 = N1 + 1
        
        for _var in (_U0, _Y0, _U1, _Y2):
            assert len(_var.shape) == 3

        self._U0 = _U0
        self._Y0 = _Y0
        self._U1 = _U1
        self._Y2 = _Y2
