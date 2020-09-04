'''
Created on 2020/07/11

@author: ukai
'''
import os
import unittest

from builder import Builder
from loader import Loader
from mylogger import MyLogger
from sid_agent import SidAgent
from sid_agent_factory import SidAgentFactory
from sid_build_parameter import SidBuildParameter
from sid_build_parameter_factory import SidBuildParameterFactory
from sid_environment_factory import SidEnvironmentFactory
from sid_trainer_factory import SidTrainerFactory
from store import Store
from sid_evaluator import SidEvaluator
from builtins import isinstance


class Test(unittest.TestCase):
    
    
    @classmethod
    def setUpClass(cls):
        super(Test, cls).setUpClass()
        
        cls.dbPath = "testDb.sqlite"
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)
    
    def setUp(self):
        unittest.TestCase.setUp(self)
        
        agentFactory = SidAgentFactory()
        environmentFactory = SidEnvironmentFactory()
        trainerFactory = SidTrainerFactory()        
        buildParameterFactory = SidBuildParameterFactory()
        store = Store(self.dbPath)
        logger = MyLogger(console_print=True)
        
        self.builder = Builder(trainerFactory, agentFactory, environmentFactory, store, logger)
        
        self.buildParameters = []
        for k1 in range(2):
            nIntervalSave = 3
            nEpoch = 5
            self.buildParameters.append(SidBuildParameter(nIntervalSave=nIntervalSave
                                                          , Ntrain=2**7
                                                          , nEpoch=nEpoch
                                                          , label="test" + str(k1)))

        for k1 in range(2):
            nIntervalSave = 3
            nEpoch = 5
            self.buildParameters.append(SidBuildParameter(nIntervalSave=nIntervalSave
                                                          , Ntrain=2**7
                                                          , nEpoch=nEpoch                                                          
                                                          , label="test" + str(k1)
                                                          , environmentClass="SidEnvironmentImbalancedSampling"))            
        
        self.loader = Loader(agentFactory, buildParameterFactory, environmentFactory, trainerFactory, store)
        
    @classmethod
    def tearDownClass(cls):
        super(Test, cls).tearDownClass()
        cls.dbPath = "testDb.sqlite"
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)

    def test001(self):
        
        evaluator = SidEvaluator()
        
        for buildParameter in self.buildParameters:
            assert isinstance(buildParameter, SidBuildParameter)
            self.builder.build(buildParameter)
            
        assert isinstance(self.loader, Loader)        
        for agent, buildParameter, epoch, environment, trainer in self.loader.load("test%", None):
            assert isinstance(agent, SidAgent)  
            assert isinstance(buildParameter, SidBuildParameter)
            
            row = evaluator.evaluate(agent, buildParameter, epoch, environment, trainer)
            
            assert isinstance(row, dict)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()