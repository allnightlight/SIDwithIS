'''
Created on 2020/09/05

@author: ukai
'''
from cs01_build_parameter import Cs01BuildParameter
from sid_build_parameter_factory import SidBuildParameterFactory


class Cs01BuildParameterFactory(SidBuildParameterFactory):
    '''
    classdocs
    '''


    def create(self):
        return Cs01BuildParameter()    