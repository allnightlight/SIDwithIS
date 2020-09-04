'''
Created on 2020/07/16

@author: ukai
'''
import unittest

from data_generator_abstract_singleton import DataGeneratorAbstractSingleton
import numpy as np
from sid_agent import SidAgent
from sid_agent001 import SidAgent001
from sid_batch_data_agent import SidBatchDataAgent
from sid_environment_abstract import SidEnvironmentAbstract
from sid_environment_normal_sampling import SidEnvironmentNormalSampling


class Test(unittest.TestCase):


    def test001(self):
        
        Ny = 2
        Nu = 3
        
        dataGeneratorSingleton = DataGeneratorAbstractSingleton(2**10, 2, 3)
        environment = SidEnvironmentNormalSampling(dataGeneratorSingleton, Ntrain=2**9, Nbatch=2**5, N0=2**2, N1=2**2)
        assert isinstance(environment, SidEnvironmentAbstract)
        
        params1 = dict(Ny=Ny, Nu=Nu, Nhidden=2**3, use_offset_compensate = True)
        params2 = dict(Ny=Ny, Nu=Nu, Nhidden=2**3, use_offset_compensate = False)
        
        for agent in (SidAgent001(**params1), SidAgent001(**params2)):
            
            for batchDataIn in environment.generateBatchDataIterator():
                batchDataOut = agent(batchDataIn)
                
                assert isinstance(batchDataOut, SidBatchDataAgent)
        

    def test002(self):
        
        Ny = 2
        Nu = 3
        
        params = dict(Ny=Ny, Nu=Nu, Nhidden=2**3, use_offset_compensate = False)
        
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