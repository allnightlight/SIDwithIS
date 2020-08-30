'''
Created on 2020/07/16

@author: ukai
'''
from sid_build_parameter import SidBuildParameter
from sid_environment import SidEnvironment
from sl_environment_factory import SlEnvironmentFactory


class SidEnvironmentFactory(SlEnvironmentFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter):
        
        assert isinstance(buildParameter, SidBuildParameter)
        
        environment = SidEnvironment(Nhidden = buildParameter.NhiddenEnv
                        , Ntrain = buildParameter.Ntrain
                        , T0 = buildParameter.T0
                        , T1 = buildParameter.T1
                        , Ny = buildParameter.Ny
                        , Nu = buildParameter.Nu
                        , Nbatch = buildParameter.Nbatch
                        , N0 = buildParameter.N0
                        , N1 = buildParameter.N1
                        , seed = buildParameter.seed)
        
        return environment