'''
Created on 2020/07/16

@author: ukai
'''
from batch_data_agent import BatchDataAgent



class SidBatchDataAgent(BatchDataAgent):
    '''
    classdocs
    '''


    def __init__(self, _Yhat, T):
        # _Y: (N2 = 1 + N1, *, Ny), T: (N2, *)
        assert len(_Yhat.shape) == 3
        
        self._Yhat = _Yhat
        self.T = T