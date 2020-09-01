'''
Created on 2020/07/10

@author: ukai
'''
import os
import unittest
import pandas as pd

from agent import Agent
from agent_factory import AgentFactory
from build_parameter import BuildParameter
from build_parameter_factory import BuildParameterFactory
from loader import Loader
from store import Store
from store_field import StoreField
from environment_factory import EnvironmentFactory
from trainer_factory import TrainerFactory
from builtins import isinstance
from environment import Environment
from trainer import Trainer
from evaluator import Evaluator


class Test(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        super(Test, cls).setUpClass()
        
        dbPath = "testDb.sqlite"
        if os.path.exists(dbPath):
            os.remove(dbPath)
        
        store = Store(dbPath)
        assert isinstance(store, Store)
        
        for k1 in range(2**3):
            buildParameter = BuildParameter(label = "test" + str(k1))
            agent = Agent()

            for epoch in range(2**4):        
        
                agentMemento = agent.createMemento()
                buildParameterMemento = buildParameter.createMemento()
                buildParameterKey = buildParameter.key
                buildParameterLabel = buildParameter.label
        
                storeField = StoreField(agentMemento, epoch, buildParameterMemento, buildParameterKey, buildParameterLabel)
                assert isinstance(storeField, StoreField)
                
                store.append(storeField)
                
        cls.dbPath = dbPath
        

    @classmethod
    def tearDownClass(cls):
        super(Test, cls).tearDownClass()
        if os.path.exists(cls.dbPath):
            os.remove(cls.dbPath)


    def test0001(self):
        
        store = Store(self.dbPath)
        agentFactory = AgentFactory()
        environmentFactory = EnvironmentFactory()
        buildParameterFactory = BuildParameterFactory()
        trainerFactory = TrainerFactory()
        
        loader = Loader(agentFactory, buildParameterFactory, environmentFactory, trainerFactory, store)
        assert isinstance(loader, Loader)
        
        for agent, buildParameter, epoch, environment, trainer in loader.load("test%", None):
            assert isinstance(agent, Agent)
            assert isinstance(buildParameter, BuildParameter)
            assert isinstance(environment, Environment)
            assert isinstance(trainer, Trainer)

        epochGiven = 1
        for agent, buildParameter, epoch, environment, trainer in loader.load("test%", epoch=epochGiven):
            assert isinstance(agent, Agent)
            assert isinstance(buildParameter, BuildParameter)
            assert epoch == epochGiven
            assert isinstance(environment, Environment)
            assert isinstance(trainer, Trainer)


        buildParameterKeyGiven = buildParameter.key
        for agent, buildParameter, epoch, environment, trainer in loader.load("test%", buildParameterKey=buildParameterKeyGiven):
            assert isinstance(agent, Agent)
            assert isinstance(buildParameter, BuildParameter)
            assert buildParameter.key == buildParameterKeyGiven
            assert isinstance(environment, Environment)
            assert isinstance(trainer, Trainer)


        for agent, buildParameter, epoch, environment, trainer in loader.load("test%", buildParameterKey=buildParameterKeyGiven, epoch = epochGiven):
            assert isinstance(agent, Agent)
            assert isinstance(buildParameter, BuildParameter)
            assert buildParameter.key == buildParameterKeyGiven
            assert epoch == epochGiven
            assert isinstance(environment, Environment)
            assert isinstance(trainer, Trainer)
            
    def test002(self):
        
        store = Store(self.dbPath)
        agentFactory = AgentFactory()
        environmentFactory = EnvironmentFactory()
        buildParameterFactory = BuildParameterFactory()
        trainerFactory = TrainerFactory()
        
        evaluator = Evaluator()
        
        loader = Loader(agentFactory, buildParameterFactory, environmentFactory, trainerFactory, store)
        assert isinstance(loader, Loader)

        tbl = {"criteria": [], "score": []}
        for agent, buildParameter, epoch, environment, trainer in loader.load("test%", None):
            row = evaluator.evaluate(agent, buildParameter, epoch, environment, trainer)
            for criteria in row:
                tbl["criteria"].append(criteria)
                tbl["score"].append(row[criteria])        
        tbl = pd.DataFrame(tbl)
        
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test0001']
    unittest.main()