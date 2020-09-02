'''
Created on 2020/07/16

@author: ukai
'''
from sl_build_parameter import SlBuildParameter

class SidBuildParameter(SlBuildParameter):
    '''
    classdocs
    '''


    def __init__(self, nIntervalSave=2 ** 4, nEpoch=2 ** 7, label="None", Ny = 1, Nu = 1, Nw=1, prob_step=1/2**5, sampling_balance = 0.5, amp_dv = 1.0, environmentClass = "SidEnvironment"
                 , NhiddenAgent=2**2, agentClass = "agent001", Nbatch = 2**5, N0=1, N1=1, seed=0, NhiddenEnv = 2**2, Ntrain = 2**12, Ntest = 2**7, T0 = 2**1, T1=2**7, dampingConstantInitial=0.99):
        SlBuildParameter.__init__(self, nIntervalSave=nIntervalSave, nEpoch=nEpoch, label=label)
        
        self.Ny = Ny
        self.Nu = Nu
        self.Nw = Nw
        self.prob_step = prob_step
        self.sampling_balance = sampling_balance
        self.amp_dv = amp_dv
        self.environmentClass = environmentClass
        self.NhiddenAgent = NhiddenAgent
        self.agentClass = agentClass
        self.Nbatch = Nbatch
        self.N0 = N0
        self.N1 = N1
        self.seed = seed
        self.NhiddenEnv = NhiddenEnv
        self.Ntrain = Ntrain
        self.Ntest = Ntest
        self.T0 = T0
        self.T1 = T1
        self.dampingConstantInitial = dampingConstantInitial