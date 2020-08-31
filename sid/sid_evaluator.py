'''
Created on 2020/08/31

@author: ukai
'''
import numpy as np

from sl_evaluator import SlEvaluator
from sid_batch_data_environment import SidBatchDataEnvironment
from sid_batch_data_agent import SidBatchDataAgent

class SidEvaluator(SlEvaluator):
    '''
    classdocs
    '''

    names = [
        "error_initial"
        , "error_first_step"
        , "error_last_step"
        , "error_median"
        ]
    
    
    def evaluateError(self, testBatchDataIn, testBatchDataOut):
        
        assert isinstance(testBatchDataIn, SidBatchDataEnvironment)
        assert isinstance(testBatchDataOut, SidBatchDataAgent)
        
        Y2 = testBatchDataIn._Y2.data.numpy() # (N2, *, Ny)
        Y2hat = testBatchDataOut._Yhat.data.numpy() # (N2, *, Ny)
        
        error = np.mean((Y2-Y2hat)**2, axis=(1,2)) # (N2,)
        row = [
            error[0]
            , error[1] if len(error) >= 1 else np.nan
            , error[-1]
            , np.median(error)
            ]
        
        return row