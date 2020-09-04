'''
Created on 2020/09/01

@author: ukai
'''
from sl_environment import SlEnvironment
from builtins import isinstance
from data_generator_abstract_singleton import DataGeneratorAbstractSingleton

class SidEnvironmentAbstract(SlEnvironment):
    '''
    classdocs
    '''

    def __init__(self, dataGeneratorSingleton, Ntrain, Nbatch, N0, N1, sampling_balance):
        SlEnvironment.__init__(self)
        
        assert isinstance(dataGeneratorSingleton, DataGeneratorAbstractSingleton)
        
        self.Ny = dataGeneratorSingleton.Ny
        self.Nu = dataGeneratorSingleton.Nu
        
        self.Ntrain = Ntrain
        self.Nbatch = Nbatch
        self.N0 = N0 # estimation horizon's length
        self.N1 = N1 # prediction horizon's length
        self.sampling_balance = sampling_balance # the proportional rate of samples with ev == 1 in a batch data