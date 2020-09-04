'''
Created on 2020/09/04

@author: ukai
'''
from sid_build_parameter import SidBuildParameter

class Cs02BuildParameter(SidBuildParameter):
    '''
    classdocs
    '''


    def __init__(self, 
        nIntervalSave=2 ** 4, 
        nEpoch=2 ** 7, 
        label="None", 
        sampling_balance=0.5, 
        NhiddenAgent=2, 
        agentClass="agent001", 
        Nbatch=2 ** 5, 
        N0=2 ** 0, 
        N1=2 ** 0, 
        Ntrain=2 ** 7, 
        dampingConstantInitial=0.99, 
        use_offset_compensate=False, 
        use_imbalanced_sampling=False,
        dataFilePath="data.csv"):
        SidBuildParameter.__init__(self, nIntervalSave=nIntervalSave, nEpoch=nEpoch, label=label, sampling_balance=sampling_balance, NhiddenAgent=NhiddenAgent, agentClass=agentClass, Nbatch=Nbatch, N0=N0, N1=N1, Ntrain=Ntrain, dampingConstantInitial=dampingConstantInitial, use_offset_compensate=use_offset_compensate, use_imbalanced_sampling=use_imbalanced_sampling)
        
        self.dataFilePath = dataFilePath