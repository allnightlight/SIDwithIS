'''
Created on 2020/08/31

@author: ukai
'''
from evaluator import Evaluator
from sl_agent import SlAgent
from builtins import isinstance
from sl_environment import SlEnvironment

class SlEvaluator(Evaluator):
    '''
    classdocs
    '''

    # <<public>> <<final?>>
    def evaluate(self, agent, buildParameter, epoch, environment, trainer):
        
        assert isinstance(agent, SlAgent)
        assert isinstance(environment, SlEnvironment)
        
        testBatchDataIn = environment.getTestBatchData()
        testBatchDataOut = agent.forward(testBatchDataIn)
        
        return self.evaluateError(testBatchDataIn, testBatchDataOut)
        
    # <<protected>>
    def evaluateError(self, testBatchDataIn, testBatchDataOut):
        return dict(zip(["RMSE", "CORR"], [1.23, 0.56,]))