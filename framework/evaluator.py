'''
Created on 2020/08/31

@author: ukai
'''

# designed as "strategy"
class Evaluator(object):
    '''
    classdocs
    '''

    
    def evaluate(self, agent, buildParameter, epoch, environment, trainer):
        
        return dict(zip(["RMSE", "CORR"], [1.23, 0.56,]))