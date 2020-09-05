'''
Created on 2020/09/05

@author: ukai
'''
from builtins import isinstance

from cs01_build_parameter import Cs01BuildParameter
from cs01_data_generator_singleton import Cs01DataGeneratorSingleton
from sid_environment_factory import SidEnvironmentFactory


class Cs01EnvironmentFactory(SidEnvironmentFactory):
    '''
    classdocs
    '''


    def createDataGeneratorSingleton(self, buildParameter):
        assert isinstance(buildParameter, Cs01BuildParameter)
        return Cs01DataGeneratorSingleton.getInstance(seed = buildParameter.seed)