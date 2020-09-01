'''
Created on 2020/07/16

@author: ukai
'''
from sid_build_parameter import SidBuildParameter
from sid_environment import SidEnvironment
from sl_environment_factory import SlEnvironmentFactory
from sid_environment_imbalanced_sampling import SidEnvironmentImbalancedSampling


class SidEnvironmentFactory(SlEnvironmentFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter):
        
        assert isinstance(buildParameter, SidBuildParameter)

        if buildParameter.environmentClass == "SidEnvironment":        
            environment = SidEnvironment(Nhidden = buildParameter.NhiddenEnv
                            , Ntrain = buildParameter.Ntrain
                            , Ntest = buildParameter.Ntest
                            , T0 = buildParameter.T0
                            , T1 = buildParameter.T1
                            , Ny = buildParameter.Ny
                            , Nu = buildParameter.Nu
                            , Nbatch = buildParameter.Nbatch
                            , N0 = buildParameter.N0
                            , N1 = buildParameter.N1
                            , seed = buildParameter.seed)

        if buildParameter.environmentClass == "SidEnvironmentImbalancedSampling":                       
            environment = SidEnvironmentImbalancedSampling(Nhidden = buildParameter.NhiddenEnv
                , Ntrain = buildParameter.Ntrain
                , Ntest = buildParameter.Ntest
                , T0 = buildParameter.T0
                , T1 = buildParameter.T1
                , Ny = buildParameter.Ny
                , Nu = buildParameter.Nu
                , Nw = buildParameter.Nw
                , prob_step = buildParameter.prob_step
                , Nbatch = buildParameter.Nbatch
                , N0 = buildParameter.N0
                , N1 = buildParameter.N1
                , sampling_balance = buildParameter.sampling_balance
                , seed = buildParameter.seed)
        
        return environment