'''
Created on 2020/08/31

@author: ukai
'''

# designed as "strategy"
class Evaluator(object):
    '''
    classdocs
    '''

    names = ["RMSE", "CORR"]
    
    
    def evaluate(self, agent, buildParameter, epoch, environment, trainer):
        return [1.23, 0.56,]