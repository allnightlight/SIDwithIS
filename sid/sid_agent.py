'''
Created on 2020/07/17

@author: ukai
'''
import os

import torch

from sl_agent import SlAgent
from util import Utils


# <<abstract>>
class SidAgent(SlAgent):
    '''
    classdocs
    '''

    checkPointPath = "./checkpoint"
    
    def createMemento(self):
        
        filename = Utils.generateRandomString(16) + ".pt"
        path = os.path.join(self.checkPointPath, filename)

        if not os.path.exists(self.checkPointPath):
            os.mkdir(self.checkPointPath)        
        torch.save(self.state_dict(), path)
        
        agentMemento = path
        
        return agentMemento
    
    def loadMemento(self, agentMemento):
        
        path = agentMemento
        self.load_state_dict(torch.load(path))
