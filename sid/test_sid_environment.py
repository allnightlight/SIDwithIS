'''
Created on 2020/07/16

@author: ukai
'''
import unittest

from data_generator_abstract_singleton import DataGeneratorAbstractSingleton
from sid_batch_data_environment import SidBatchDataEnvironment
from sid_environment_abstract import SidEnvironmentAbstract
from sid_environment_normal_sampling import SidEnvironmentNormalSampling
from sid_environment_imbalanced_sampling import SidEnvironmentImbalancedSampling


class Test(unittest.TestCase):


    def test001(self):
        
        for _ in range(10):
            dg = DataGeneratorAbstractSingleton.getInstance(Nsample=2**10, Ny=2, Nu=3)
            dg.loadData()
    
        
    def test002(self):
        
        dataGeneratorSingleton = DataGeneratorAbstractSingleton(2**10, 2, 3)
        dataGeneratorSingleton.loadData()
        environment = SidEnvironmentAbstract(dataGeneratorSingleton, Ntrain=2**9, Nbatch=2**5, N0=2**2, N1=2**2)
        assert isinstance(environment, SidEnvironmentAbstract)


    def test003(self):
        
        dataGeneratorSingleton = DataGeneratorAbstractSingleton(2**10, 2, 3)
        dataGeneratorSingleton.loadData()
        environment = SidEnvironmentNormalSampling(dataGeneratorSingleton, Ntrain=2**9, Nbatch=2**5, N0=2**2, N1=2**2)
        assert isinstance(environment, SidEnvironmentAbstract)
        
        for batchDataEnvironment in environment.generateBatchDataIterator():
            assert isinstance(batchDataEnvironment, SidBatchDataEnvironment)

    def test004(self):
        
        dataGeneratorSingleton = DataGeneratorAbstractSingleton(2**10, 2, 3)
        dataGeneratorSingleton.loadData()
        environment = SidEnvironmentImbalancedSampling(dataGeneratorSingleton, Ntrain=2**9, Nbatch=2**5, N0=2**2, N1=2**2, sampling_balance=0.5)
        assert isinstance(environment, SidEnvironmentAbstract)
        
        for batchDataEnvironment in environment.generateBatchDataIterator():
            assert isinstance(batchDataEnvironment, SidBatchDataEnvironment)

    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()