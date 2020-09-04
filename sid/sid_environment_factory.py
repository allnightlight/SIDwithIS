'''
Created on 2020/07/16

@author: ukai
'''
from sid_build_parameter import SidBuildParameter
from sid_environment import SidEnvironment
from sl_environment_factory import SlEnvironmentFactory
from sid_environment_imbalanced_sampling import SidEnvironmentImbalancedSampling
from sid_environment_abstract import SidEnvironmentAbstract
from data_generator_abstract_singleton import DataGeneratorAbstractSingleton


class SidEnvironmentFactory(SlEnvironmentFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter):
        
        assert isinstance(buildParameter, SidBuildParameter)
        
        dataGeneratorSingleton = DataGeneratorAbstractSingleton(Nsample=2**10, Ny=2, Nu=3)
        environment = SidEnvironmentAbstract(dataGeneratorSingleton
                               , Ntrain = buildParameter.Ntrain
                               , Nbatch = buildParameter.Nbatch
                               , N0 = buildParameter.N0
                               , N1 = buildParameter.N1
                               , sampling_balance = buildParameter.sampling_balance)
        
        return environment