'''
Created on 2020/09/03

@author: ukai
'''

import numpy as np
from data_generator_abstract_singleton import DataGeneratorAbstractSingleton


class DataGeneratorFromCsvSingleton(DataGeneratorAbstractSingleton):
    '''
    classdocs
    '''
    
    # <<private>>
    def __init__(self, dataFilePath):
        Ev, U, Y, Ntrain, Ntest = self.loadData(dataFilePath)
        
        self.Ev = Ev # (N,)
        self.U = U # (N, Nu)
        self.Y = Y # (N, Ny)
        self.Ntrain = Ntrain
        self.Ntest = Ntest    
    
    # <<private>>
    def loadData(self, dataFilePath):
        with open(dataFilePath, "r") as fp:
            lines = fp.readlines()
    
        header = lines[0].rstrip().split(",")
        Nu = sum([1 for elm in header if elm[0] == "U"])
        Ny = sum([1 for elm in header if elm[0] == "Y"])
    
        Ev = []
        Segment = []
        U = []
        Y = []
    
        for line in lines[1:]:
            row = line.rstrip().split(",")
            Segment.append(int(row[0]))
            Ev.append(int(row[1]))
            U.append( [float(row[2+k1]) for k1 in range(Nu) ]  )
            Y.append( [float(row[2+Nu+k1]) for k1 in range(Ny) ]  )
    
        U = np.array(U) # (N, Nu)
        Y = np.array(Y) # (N, Ny)
        Ev = np.array(Ev) # (N,)
        Segment = np.array(Segment) # (N,)
        Ntrain = sum(Segment == 0)
        Ntest = sum(Segment == 1)
        
        return Ev, U, Y, Ntrain, Ntest 
