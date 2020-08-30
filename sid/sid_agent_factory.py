'''
Created on 2020/07/16

@author: ukai
'''
from sid_agent001 import SidAgent001
from sid_build_parameter import SidBuildParameter
from sl_agent_factory import SlAgentFactory


class SidAgentFactory(SlAgentFactory):
    '''
    classdocs
    '''


    def create(self, buildParameter, environment):
        
        assert isinstance(buildParameter, SidBuildParameter)
        
        agent = None
        if buildParameter.agentClass == "agent001":
            agent = SidAgent001(Ny = buildParameter.Ny, Nu = buildParameter.Nu, Nhidden = buildParameter.NhiddenAgent)
        assert agent is not None
        
        return agent