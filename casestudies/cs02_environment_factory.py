'''
Created on 2020/09/04

@author: ukai
'''
from cs02_data_generator_singleton import Cs02DataGeneratorSingleton
from sid_environment_factory import SidEnvironmentFactory
from builtins import isinstance
from cs02_build_parameter import Cs02BuildParameter


class Cs02EnvironmentFactory(SidEnvironmentFactory):
    '''
    classdocs
    '''


    def createDataGeneratorSingleton(self, buildParameter):
        assert isinstance(buildParameter, Cs02BuildParameter)
        return Cs02DataGeneratorSingleton.getInstance(dataFilePath=buildParameter.dataFilePath)