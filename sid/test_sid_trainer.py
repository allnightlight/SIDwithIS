'''
Created on 2020/07/16

@author: ukai
'''
from builtins import isinstance
import unittest

from data_generator_abstract_singleton import DataGeneratorAbstractSingleton
from sid_agent001 import SidAgent001
from sid_environment_abstract import SidEnvironmentAbstract
from sid_trainer import SidTrainer
from sid_environment_normal_sampling import SidEnvironmentNormalSampling


class Test(unittest.TestCase):


    def test001(self):
        
        Nsample = 2**10
        Ny = 2
        Nu = 3
        
        dataGeneratorSingleton = DataGeneratorAbstractSingleton.getInstance(Nsample=2**10, Ny=2, Nu=3)
        dataGeneratorSingleton.loadData()
        environment = SidEnvironmentNormalSampling(dataGeneratorSingleton, Ntrain=2**9, Nbatch=2**5, N0=2**2, N1=2**2)
        assert isinstance(environment, SidEnvironmentAbstract)

        
        params = dict(Ny=Ny, Nu=Nu, Nhidden=2**3, use_offset_compensate = True)
        
        for agent in (SidAgent001(**params),):
            
            trainer = SidTrainer(agent, environment)
            assert isinstance(trainer, SidTrainer)
            
            trainer.train()
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()