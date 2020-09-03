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
    
    
    def evaluateError(self, testBatchDataIn, testBatchDataOut):
        
        assert isinstance(testBatchDataIn, SidBatchDataEnvironment)
        assert isinstance(testBatchDataOut, SidBatchDataAgent)
        
        Y2 = testBatchDataIn._Y2.data.numpy() # (N2, *, Ny)
        Y2hat = testBatchDataOut._Yhat.data.numpy() # (N2, *, Ny)
        
        mseForEveryStep = np.mean((Y2-Y2hat)**2, axis=(1,2)) # (N2,)
        rmseForEveryStep = np.sqrt(mseForEveryStep) # (N2,)
        rmseAverage = np.sqrt(np.mean(mseForEveryStep)) # (,)
        
        row = {}
        for k1 in range(rmseForEveryStep.shape[0]):
            row["RMSE_STEP%02d" % k1] = rmseForEveryStep[k1]
        row["RMSE_AVERAGE"] = rmseAverage
        
        return row