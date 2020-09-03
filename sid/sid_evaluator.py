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

        def getError(Y, Yhat, labelPrefix):            
            # Y, Yhat: (N2, *, Ny), labelPrefix: as string
            
            if Y.shape[1] == 0:
                return {}
            else:
                mseForEveryStep = np.mean((Y-Yhat)**2, axis=(1,2)) # (N2,)
                rmseForEveryStep = np.sqrt(mseForEveryStep) # (N2,)
                rmseAverage = np.sqrt(np.mean(mseForEveryStep)) # (,)
            
                row = {}
                for k1 in range(rmseForEveryStep.shape[0]):
                    row[labelPrefix + "_STEP%02d" % k1] = rmseForEveryStep[k1]
                row[labelPrefix + "_AVERAGE"] = rmseAverage
    
                return row
                
        Y2 = testBatchDataIn._Y2.data.numpy() # (N2, *, Ny)
        Y2hat = testBatchDataOut._Yhat.data.numpy() # (N2, *, Ny)
        Ev1 = testBatchDataIn._Ev1.data.numpy() # (N2, *)
        idxEvOn = Ev1[0, :] == 1 # (*)
        idxEvOff = Ev1[0, :] == 0 # (*)
        
        rowTotal = getError(Y2, Y2hat, "RMSE_TOTAL")
        rowEvOn = getError(Y2[:,idxEvOn,:], Y2hat[:,idxEvOn,:], "RMSE_EvOn")
        rowEvOff = getError(Y2[:,idxEvOff,:], Y2hat[:,idxEvOff,:], "RMSE_EvOff")
        
        row = {**rowTotal, **rowEvOn, **rowEvOff}        
        
        return row