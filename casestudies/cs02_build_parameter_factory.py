'''
Created on 2020/09/04

@author: ukai
'''
from sid_build_parameter_factory import SidBuildParameterFactory
from cs02_build_parameter import Cs02BuildParameter

class Cs02BuildParameterFactory(SidBuildParameterFactory):
    '''
    classdocs
    '''


    def create(self):
        return Cs02BuildParameter()    