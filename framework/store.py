'''
Created on 2020/07/09

@author: ukai
'''
import os
import sqlite3

from store_field import StoreField
import json
from util import Utils
import glob
import traceback


class Store(object):
    '''
    classdocs
    '''
    
    
    def __init__(self, dbPath, trainLogFolderPath = "./trainlog"):
                
        self.dbPath = dbPath        
        self.trainLogFolderPath = trainLogFolderPath
        if not os.path.exists(trainLogFolderPath):
            os.mkdir(trainLogFolderPath)


    def append(self, storeField):
        assert isinstance(storeField, StoreField)
        
        dataToSave = {
            "buildParameterKey": storeField.buildParameterKey
            , "buildParameterLabel": storeField.buildParameterLabel
            , "buildParameterMemento": storeField.buildParameterMemento
            , "agentMemento": storeField.agentMemento
            , "epoch": storeField.epoch   
            }
        
        trainLogFilePath = os.path.join(self.trainLogFolderPath, Utils.generateRandomString(16))        
        with open(trainLogFilePath, "w") as fp:
            json.dump(dataToSave, fp)
            
    
    def update_db(self):
        
        def create_db(dbPath):
            conn = sqlite3.connect(dbPath)
            cur = conn.cursor()
        
            cur.executescript("""
        
            Drop Table If Exists BuildParameter;
            Create Table BuildParameter(
                build_parameter_id Integer Primary Key,
                build_parameter_key Text Unique,                
                build_parameter_label Text,
                build_parameter_memento Text
            );
        
            Drop Table If Exists TrainLog;
            Create Table TrainLog(
                train_log_id Integer Primary Key,
                build_parameter_id Integer,
                agent_memento Text Unique,
                epoch Integer,
                Unique(build_parameter_id, epoch),
                FOREIGN KEY (build_parameter_id) REFERENCES BuildParameter (build_parameter_id) 
            );
        
            """)
        
            conn.close()

        def myupdate(dbPath, build_parameter_key, build_parameter_label, build_parameter_memento, agent_memento, epoch):
            conn = sqlite3.connect(dbPath)
            cur = conn.cursor()
        
            cur.execute("""
            Insert Or Ignore Into BuildParameter (
                build_parameter_key
                , build_parameter_label
                , build_parameter_memento
                ) values (?, ?, ?)
            """, (build_parameter_key, build_parameter_label, build_parameter_memento,))
            cur.execute("""
            Select 
                build_parameter_id
                    From BuildParameter
                    Where build_parameter_key = ?
            """, (build_parameter_key,))
            build_parameter_id, = cur.fetchone()
        
            cur.execute("""
            Insert Or Ignore Into TrainLog (
                build_parameter_id
                , agent_memento
                , epoch
                ) values (?,?,?)
            """, (build_parameter_id, agent_memento, epoch,))
            
            conn.commit()
            conn.close()

        if not os.path.exists(self.dbPath):
            create_db(self.dbPath)
        
        for trainLogFilePath in glob.glob(os.path.join(self.trainLogFolderPath, "*")):
            with open(trainLogFilePath, "r") as fp:
                dataFromFile = json.load(fp)
                myupdate(self.dbPath
                         , dataFromFile["buildParameterKey"]
                         , dataFromFile["buildParameterLabel"]
                         , dataFromFile["buildParameterMemento"]
                         , dataFromFile["agentMemento"]
                         , dataFromFile["epoch"])
            try:
                os.remove(trainLogFilePath)
            except :
                traceback.print_exc()


    def restore(self, buildParameterLabel, epoch = None, buildParameterKey = None):
                
        def my_find_all(dbPath, buildParameterKey, buildParameterLabel, epoch):
            conn = sqlite3.connect(dbPath)
            cur = conn.cursor()
                    
            sql = """\
            Select
                t.agent_memento
                , t.epoch
                , b.build_parameter_memento
                , b.build_parameter_key
                , b.build_parameter_label
                From TrainLog t
                    Join BuildParameter b
                        On t.build_parameter_id = b.build_parameter_id
                Where b.build_parameter_label like ?"""
            if epoch is not None:
                sql += " And epoch = %d" % epoch
            if buildParameterKey is not None:
                sql += " And build_parameter_key = \"%s\"" % buildParameterKey
            sql += " Order by b.build_parameter_id, t.epoch"
                
            cur.execute(sql, (buildParameterLabel,))
                
            res = [*cur.fetchall()]
            conn.close()

            for agent_memento, epoch, buildParameterMemento, buildParameterKey, buildParameterLabel in res:
                yield (agent_memento, epoch, buildParameterMemento, buildParameterKey, buildParameterLabel)
        
        for (agent_memento, epoch, buildParameterMemento, buildParameterKey, buildParameterLabel) in my_find_all(self.dbPath, buildParameterKey, buildParameterLabel, epoch):
            yield StoreField(agent_memento, epoch, buildParameterMemento, buildParameterKey, buildParameterLabel)