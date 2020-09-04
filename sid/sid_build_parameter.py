'''
Created on 2020/07/16

@author: ukai
'''
from sl_build_parameter import SlBuildParameter

class SidBuildParameter(SlBuildParameter):
    '''
    classdocs
    '''


    def __init__(self
                 , nIntervalSave=2 ** 4
                 , nEpoch=2 ** 7
                 , label="None"
                 , sampling_balance=0.5
                 , NhiddenAgent=2
                 , agentClass="agent001"
                 , Nbatch=2**5
                 , N0=2**0
                 , N1=2**0
                 , Ntrain=2**7
                 , dampingConstantInitial=0.99
                 , use_offset_compensate=False):        
        SlBuildParameter.__init__(self, nIntervalSave=nIntervalSave, nEpoch=nEpoch, label=label)

        self.sampling_balance = sampling_balance        
        self.NhiddenAgent = NhiddenAgent
        self.agentClass = agentClass
        self.Nbatch = Nbatch
        self.N0 = N0
        self.N1 = N1        
        self.Ntrain = Ntrain        
        self.dampingConstantInitial = dampingConstantInitial
        self.use_offset_compensate = use_offset_compensate
