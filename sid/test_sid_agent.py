'''
Created on 2020/07/16

@author: ukai
'''
import unittest

import numpy as np
from sid_agent import SidAgent
from sid_agent001 import SidAgent001
from sid_batch_data_agent import SidBatchDataAgent
from sid_environment import SidEnvironment


class Test(unittest.TestCase):


    def test001(self):
        
        Ny = 2
        Nu = 3
        
        environment = SidEnvironment(Nhidden=2**2, Ntrain=2**10, T0=2**3, T1=2**4, Ny=Ny, Nu=Nu, Nbatch=2**5, N0=2**3, N1=2**2, seed = 1)
        
        params = dict(Ny=Ny, Nu=Nu, Nhidden=2**3)
        
        for agent in (SidAgent001(**params),):
            
            for batchDataIn in environment.generateBatchDataIterator():
                batchDataOut = agent(batchDataIn)
                
                assert isinstance(batchDataOut, SidBatchDataAgent)
        

    def test002(self):
        
        Ny = 2
        Nu = 3
        
        params = dict(Ny=Ny, Nu=Nu, Nhidden=2**3)
        
        agent = SidAgent001(**params)
        agentAnother = SidAgent001(**params)
        assert isinstance(agent, SidAgent)
        
        agentMemento = agent.createMemento()
        
        agentAnother.loadMemento(agentMemento)
        
        for _p1, _p2 in zip(agent.parameters(), agentAnother.parameters()):
            p1 = _p1.data.numpy()
            p2 = _p2.data.numpy()
            assert np.all(p1 == p2)        

        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()