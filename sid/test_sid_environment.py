'''
Created on 2020/07/16

@author: ukai
'''
import unittest

from data_generator_singleton import DataGeneratorSingleton
import numpy as np
from sid_batch_data_environment import SidBatchDataEnvironment
from sid_environment import SidEnvironment
from sid_environment_imbalanced_sampling import SidEnvironmentImbalancedSampling
from data_generator_from_csv_singleton import DataGeneratorFromCsvSingleton


class Test(unittest.TestCase):


    def test001(self):
        
        for _ in range(10):
            dg = DataGeneratorSingleton.getInstance(Nhidden=2**2, Ntrain=2**10, T0=2**3, T1=2**4, Ny=1, Nu=1, seed = 0)
    
            dg = DataGeneratorSingleton.getInstance(Nhidden=2**2, Ntrain=2**10, T0=2**3, T1=2**4, Ny=1, Nu=1, seed = 1)
            
        
    def test002(self):
        
        environment = SidEnvironment(Nhidden=2**2, Ntrain=2**10, Ntest =2**5, T0=2**3, T1=2**4, Ny=1, Nu=1, Nbatch=2**5, N0=2**3, N1=2**2, seed = 1)
        for batchDataEnvironment in environment.generateBatchDataIterator():
            assert isinstance(batchDataEnvironment, SidBatchDataEnvironment)
        
    def test003(self):
        
        for _ in range(10):
            dg = DataGeneratorSingleton.getInstance(Nhidden=2**2, Ntrain=2**10, T0=2**3, T1=2**4, Ny=1, Nu=1, Nw = 0, action_distribution = "step", seed = 0)
            
    @unittest.skip("tmp")
    def test004(self):
        import matplotlib.pylab as plt
        
        dg = DataGeneratorSingleton.getInstance(Nhidden=2**2, Ntrain=2**10, T0=2**3, T1=2**4, Ny=2, Nu=2, Nw = 1, prob_step = 1/2**8, action_distribution = "step", amp_dv = 1.0, seed = 0)
        plt.plot(dg.Y)
        plt.show()

    def test005(self):
        
        Nbatch = 2**5
        N0 = 2**3
        N1 = 2**2
        for _ in range(10):
            environment = SidEnvironmentImbalancedSampling(Nhidden=2**2, Ntrain=2**10, Ntest =2**5, T0=2**3, T1=2**4, Ny=1, Nu=1, Nw=1, prob_step=1/2**3, Nbatch=Nbatch, N0=N0, N1=N1, sampling_balance=0.5, amp_dv = 1.0, seed = 1)
            for batchDataEnvironment in environment.generateBatchDataIterator():
                assert isinstance(batchDataEnvironment, SidBatchDataEnvironment)
                assert np.all(batchDataEnvironment._U0.shape[0:2] == (N0, Nbatch))
                assert np.all(batchDataEnvironment._Y0.shape[0:2] == (N0, Nbatch))
                assert np.all(batchDataEnvironment._Y2.shape[0:2] == (N1+1, Nbatch))
                assert np.all(batchDataEnvironment._U1.shape[0:2] == (N1, Nbatch))
            environment.getTestBatchData()

    def test006(self):
        
        dg = DataGeneratorFromCsvSingleton("data.csv")
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test001']
    unittest.main()