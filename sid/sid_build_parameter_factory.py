'''
Created on 2020/07/16

@author: ukai
'''
from sl_build_parameter_factory import SlBuildParameterFactory
from sid_build_parameter import SidBuildParameter

class SidBuildParameterFactory(SlBuildParameterFactory):
    '''
    classdocs
    '''


    def create(self):
        return SidBuildParameter()    