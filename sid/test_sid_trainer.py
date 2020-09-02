'''
Created on 2020/07/16

@author: ukai
'''
from builtins import isinstance
import unittest

from sid_agent001 import SidAgent001
from sid_environment import SidEnvironment
from sid_trainer import SidTrainer


class Test(unittest.TestCase):


    def test001(self):
        
        Ny = 2
        Nu = 3
        
        environment = SidEnvironment(Nhidden=2**2, Ntrain=2**10, Ntest =2**5, T0=2**3, T1=2**4, Ny=Ny, Nu=Nu, Nbatch=2**5, N0=2**3, N1=2**2, seed = 1)
        
        params = dict(Ny=Ny, Nu=Nu, Nhidden=2**3, use_offset_compensate = True)
        
        for agent in (SidAgent001(**params),):
            
            trainer = SidTrainer(agent, environment)
            assert isinstance(trainer, SidTrainer)
            
            trainer.train()
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()