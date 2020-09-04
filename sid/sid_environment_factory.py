'''
Created on 2020/07/16

@author: ukai
'''
from data_generator_abstract_singleton import DataGeneratorAbstractSingleton
from sid_build_parameter import SidBuildParameter
from sid_environment_normal_sampling import SidEnvironmentNormalSampling
from sl_environment_factory import SlEnvironmentFactory
from sid_environment_imbalanced_sampling import SidEnvironmentImbalancedSampling


class SidEnvironmentFactory(SlEnvironmentFactory):
    '''
    classdocs
    '''


    # <<public, final>>
    def create(self, buildParameter):
        
        assert isinstance(buildParameter, SidBuildParameter)
        
        dataGeneratorSingleton = self.createDataGeneratorSingleton(buildParameter)
        assert isinstance(dataGeneratorSingleton, DataGeneratorAbstractSingleton)
        
        if buildParameter.use_imbalanced_sampling:
            environment = SidEnvironmentImbalancedSampling(dataGeneratorSingleton
                                   , Ntrain = buildParameter.Ntrain
                                   , Nbatch = buildParameter.Nbatch
                                   , N0 = buildParameter.N0
                                   , N1 = buildParameter.N1
                                   , sampling_balance=buildParameter.sampling_balance)        
        else:
            environment = SidEnvironmentNormalSampling(dataGeneratorSingleton
                                   , Ntrain = buildParameter.Ntrain
                                   , Nbatch = buildParameter.Nbatch
                                   , N0 = buildParameter.N0
                                   , N1 = buildParameter.N1)
        
        return environment
    
    # <<protected>>
    def createDataGeneratorSingleton(self, buildParameter):
        dataGeneratorSingleton = DataGeneratorAbstractSingleton.getInstance(Nsample=buildParameter.Ntrain+2**7, Ny=2, Nu=3)
        return dataGeneratorSingleton