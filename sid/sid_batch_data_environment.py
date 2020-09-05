'''
Created on 2020/07/16

@author: ukai
'''
from batch_data_environment import BatchDataEnvironment

class SidBatchDataEnvironment(BatchDataEnvironment):
    '''
    classdocs
    '''


    def __init__(self, _U0, _Ev0, _Y0, _U1, _Ev1, _Y2, T0=None, T2=None):
        # _U0: (N0, *, Nu), _Ev0: (N0, *), _Y0: (N0, *, Ny), T0: (N0, *)
        # _U1: (N1, *, Nu), _Ev1: (N1, *), _Y2: (N2, *, Ny), N2 = N1 + 1, T2: (N2, *)
        
        for _var in (_U0, _Y0, _U1, _Y2):
            assert len(_var.shape) == 3
        for _var in (_Ev0, _Ev1):
            assert len(_var.shape) == 2

        self._U0 = _U0
        self._Ev0 = _Ev0
        self._Y0 = _Y0
        self._U1 = _U1
        self._Ev1 = _Ev1
        self._Y2 = _Y2
        self.T0 = T0
        self.T2 = T2
