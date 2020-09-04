'''
Created on 2020/09/03

@author: ukai
'''

import numpy as np
from data_generator_abstract_singleton import DataGeneratorAbstractSingleton


class Cs02DataGeneratorSingleton(DataGeneratorAbstractSingleton):
    '''
    classdocs
    '''
    
    # <<private>>
    def __init__(self, dataFilePath):
        DataGeneratorAbstractSingleton.__init__(self, Nsample=None, Ny=None, Nu=None)
        
        self.dataFilePath = dataFilePath
        
        self.Ev = None
        self.U = None
        self.Y = None
        self.Timestamp = None
        self.IsNaN = None
        self.Nsample = None
        self.Ny = None
        self.Nu = None
        
    # <<private>>
    def loadData(self):
        
        with open(self.dataFilePath, "r") as fp:
            lines = fp.readlines()
    
        header = lines[0].rstrip().split(",")
        Nu = sum([1 for elm in header if elm[0] == "U"])
        Ny = sum([1 for elm in header if elm[0] == "Y"])
    
        Ev = []
        Timestamp = []
        U = []
        Y = []
    
        for line in lines[1:]:
            row = line.rstrip().split(",")
            Timestamp.append(row[0])
            Ev.append(int(row[1]))
            U.append( [float(row[2+k1]) for k1 in range(Nu) ]  )
            Y.append( [float(row[2+Nu+k1]) for k1 in range(Ny) ]  )
    
        U = np.array(U) # (N, Nu)
        Y = np.array(Y) # (N, Ny)
        Ev = np.array(Ev) # (N,)
        Timestamp = np.array(Timestamp) # (N,)
        
        IsNaN = np.any(np.isnan(U), axis=-1) | np.any(np.isnan(Y), axis=-1) | np.any(np.isnan(Ev), axis=-1) # (N,)
        
        self.Ev = Ev # (N,)
        self.U = U # (N, Nu)
        self.Y = Y # (N, Ny)
        self.Timestamp = Timestamp # (N,)
        self.IsNaN = IsNaN # (N,)
        self.Nsample = Timestamp.shape[0]
        self.Ny = Y.shape[1]
        self.Nu = U.shape[1]