'''
Created on 2020/07/16

@author: ukai
'''
import unittest

from data_generator_abstract_singleton import DataGeneratorAbstractSingleton
from sid_batch_data_environment import SidBatchDataEnvironment
from sid_environment_abstract import SidEnvironmentAbstract
from sid_environment_normal_sampling import SidEnvironmentNormalSampling


class Test(unittest.TestCase):


    def test001(self):
        
        for _ in range(10):
            dg = DataGeneratorAbstractSingleton.getInstance(Nsample=2**10, Ny=2, Nu=3)
    
        
    def test002(self):
        
        dataGeneratorSingleton = DataGeneratorAbstractSingleton(2**10, 2, 3)
        environment = SidEnvironmentAbstract(dataGeneratorSingleton, Ntrain=2**9, Nbatch=2**5, N0=2**2, N1=2**2)
        assert isinstance(environment, SidEnvironmentAbstract)


    def test003(self):
        
        dataGeneratorSingleton = DataGeneratorAbstractSingleton(2**10, 2, 3)
        environment = SidEnvironmentNormalSampling(dataGeneratorSingleton, Ntrain=2**9, Nbatch=2**5, N0=2**2, N1=2**2)
        assert isinstance(environment, SidEnvironmentAbstract)
        
        for batchDataEnvironment in environment.generateBatchDataIterator():
            assert isinstance(batchDataEnvironment, SidBatchDataEnvironment)

    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()