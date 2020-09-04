'''
Created on 2020/07/16

@author: ukai
'''
from data_generator_abstract_singleton import DataGeneratorAbstractSingleton
from sid_build_parameter import SidBuildParameter
from sid_environment_normal_sampling import SidEnvironmentNormalSampling
from sl_environment_factory import SlEnvironmentFactory


class SidEnvironmentFactory(SlEnvironmentFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter):
        
        assert isinstance(buildParameter, SidBuildParameter)
        
        dataGeneratorSingleton = DataGeneratorAbstractSingleton(Nsample=2**10, Ny=2, Nu=3)
        environment = SidEnvironmentNormalSampling(dataGeneratorSingleton
                               , Ntrain = buildParameter.Ntrain
                               , Nbatch = buildParameter.Nbatch
                               , N0 = buildParameter.N0
                               , N1 = buildParameter.N1)
        
        return environment