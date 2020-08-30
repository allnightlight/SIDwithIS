'''
Created on 2020/07/16

@author: ukai
'''
from sid_trainer import SidTrainer
from sl_trainer_factory import SlTrainerFactory


class SidTrainerFactory(SlTrainerFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter, agent, environment):
        trainer = SidTrainer(agent, environment)
        return trainer