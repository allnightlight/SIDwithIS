'''
Created on 2020/07/11

@author: ukai
'''
import os
import unittest
import numpy as np

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
    def generateTestData(cls):
        
        Nsample = 2**10
        Nu = 2
        Ny = 3
        
        Timestamp = np.arange(Nsample)
        U = np.random.randn(Nsample, Nu)
        Y = np.random.randn(Nsample, Ny)
        Ev = np.random.randint(2, size=(Nsample,))
        
        Y[np.random.rand(Nsample, Ny) < 1/2**4] = np.nan
        
        tbl = []
        header = ["timestamp", "event",] + ["U%d" % k1 for k1 in range(Nu)] + ["Y%d" % k1 for k1 in range(Ny)]
        tbl.append(header)
        for seg, ev, u, y in zip(Timestamp, Ev.astype(np.int), U, Y):    
            row = [seg, ev, *u, *y]
            tbl.append(row)
        
        with open(cls.dataFilePath, "w") as fp:
            fp.write(",".join(header)+"\n")
            for line in tbl[1:]:
                fp.write(",".join([str(elm) for elm in line])+"\n")
    
    @classmethod
    def setUpClass(cls):
        super(Test, cls).setUpClass()
        
        cls.dbPath = "testDb.sqlite"
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)
        cls.dataFilePath = "test_data.csv"
        cls.generateTestData()
        
    def setUp(self):
        unittest.TestCase.setUp(self)
        
        agentFactory = SidAgentFactory()
        environmentFactory = Cs02EnvironmentFactory()
        trainerFactory = SidTrainerFactory()
        buildParameterFactory = Cs02BuildParameterFactory()
        store = Store(self.dbPath, trainLogFolderPath="trainLogCs02")
        logger = MyLogger(console_print=True)
        
        self.builder = Builder(trainerFactory, agentFactory, environmentFactory, store, logger)
        
        self.buildParameters = []
        for k1 in range(2):
            nIntervalSave = 3
            nEpoch = 5
            self.buildParameters.append(Cs02BuildParameter(nIntervalSave=nIntervalSave
                                                          , Ntrain=2**7
                                                          , nEpoch=nEpoch
                                                          , label="test" + str(k1)
                                                          , dataFilePath = self.dataFilePath))
            
            self.buildParameters.append(Cs02BuildParameter(nIntervalSave=nIntervalSave
                                              , Ntrain=2**7
                                              , nEpoch=nEpoch
                                              , label="test imbalanced sampling" + str(k1)
                                              , use_imbalanced_sampling = True
                                              , dataFilePath = self.dataFilePath))

        
        self.loader = Loader(agentFactory, buildParameterFactory, environmentFactory, trainerFactory, store)
        
    @classmethod
    def tearDownClass(cls):
        super(Test, cls).tearDownClass()
        cls.dbPath = "testDb.sqlite"
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)
        if os.path.exists(cls.dataFilePath):
            os.remove(cls.dataFilePath)

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