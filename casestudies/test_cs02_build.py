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
from sid_trainer_factory import SidTrainerFactory
from store import Store
from sid_evaluator import SidEvaluator
from builtins import isinstance
from cs02_environment_factory import Cs02EnvironmentFactory
from cs02_build_parameter_factory import Cs02BuildParameterFactory
from cs02_build_parameter import Cs02BuildParameter


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
        environmentFactory = Cs02EnvironmentFactory()
        trainerFactory = SidTrainerFactory()
        buildParameterFactory = Cs02BuildParameterFactory()
        store = Store(self.dbPath)
        logger = MyLogger(console_print=True)
        
        self.builder = Builder(trainerFactory, agentFactory, environmentFactory, store, logger)
        
        self.buildParameters = []
        for k1 in range(2):
            nIntervalSave = 3
            nEpoch = 5
            self.buildParameters.append(Cs02BuildParameter(nIntervalSave=nIntervalSave
                                                          , Ntrain=2**7
                                                          , nEpoch=nEpoch
                                                          , label="test" + str(k1)))
            
            self.buildParameters.append(Cs02BuildParameter(nIntervalSave=nIntervalSave
                                              , Ntrain=2**7
                                              , nEpoch=nEpoch
                                              , label="test imbalanced sampling" + str(k1)
                                              , use_imbalanced_sampling = True))

        
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