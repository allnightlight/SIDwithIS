'''
Created on 2020/09/06

@author: ukai
'''

import os

import matplotlib.pylab as plt
import numpy as np
from sid_batch_data_agent import SidBatchDataAgent
from sid_batch_data_environment import SidBatchDataEnvironment
from sl_evaluator import SlEvaluator
from util import Utils


class SidEvaluatorTrendViewer(SlEvaluator):
    '''
    classdocs
    '''
    
    figFolderPath = "./fig"
    
    def __init__(self, Nfig, figSize = [10, 6]):
        SlEvaluator.__init__(self)
        
        self.Nfig = Nfig
        self.figSize = figSize
    
    def draw_trend(self, T0, Y0, U0, T2, Y2, U1, Y2hat):
        
        # T0: (N0,), Y0: (N0, Ny), U0: (N0, Nu)
        # T2: (N1+1,), Y2: (N1+1,Ny), U1: (N1,Nu), Y2hat: (N1+1, Ny)

        T1 = T2[1:]
        N0, Ny = Y0.shape
        N1, _ = U1.shape
        _, Nu = U0.shape
    
        fig = plt.gcf()
        
        ax = fig.add_subplot(2,1,1)
        for k1 in range(Ny):
            lineSpec, = ax.plot(Y2[:,k1], linestyle = "-", label = "Y{0}".format(k1))
            ax.plot(Y2hat[:,k1], color = lineSpec.get_color(), linestyle="--")
        ax.set_xticks((0, N1))
        ax.set_xticklabels([T2[0], T2[-1]])
        ax.legend()
        ax.grid()
    
        U2 = np.concatenate((U0[-1,None,:], U1), axis=0)
    
        ax = fig.add_subplot(2,1,2)
        for k1 in range(Nu):
            ax.plot(U2[:,k1], label = "U{0}".format(k1))
        ax.set_xticks((0, N1))
        ax.set_xticklabels([T2[0], T2[-1]])
        ax.legend()
        ax.grid()
    
        fig.tight_layout()
        
    
    def evaluateError(self, testBatchDataIn, testBatchDataOut):
        
        assert isinstance(testBatchDataIn, SidBatchDataEnvironment)
        assert isinstance(testBatchDataOut, SidBatchDataAgent)
        
        T0 = testBatchDataIn.T0 # (N0, *)
        Y0 = testBatchDataIn._Y0.data.numpy() # (N0, *, Ny)
        U0 = testBatchDataIn._U0.data.numpy() # (N0, *, Nu)        
        T2 = testBatchDataIn.T2 # (N2, *)
        U1 = testBatchDataIn._U1.data.numpy() # (N1, *, Nu)
        Y2 = testBatchDataIn._Y2.data.numpy() # (N2, *, Ny)
        Y2hat = testBatchDataOut._Yhat.data.numpy() # (N2, *, Ny)
        
        Ev1 = testBatchDataIn._Ev1.data.numpy() # (N1, *)
        idxEvOn = np.where(Ev1[0,:] == 1)[0]
        
        if not os.path.exists(self.figFolderPath):
            os.mkdir(self.figFolderPath)
        
        prefix = Utils.generateRandomString(16)
        figFilePathFormat = os.path.join(self.figFolderPath, "%s_current_time={0}.png" % prefix) 
        for k1 in np.random.permutation(idxEvOn)[:np.min((self.Nfig, len(idxEvOn)))]:
                    
            fig = plt.figure(figsize=self.figSize)
            self.draw_trend(T0[:,k1], Y0[:,k1,:], U0[:,k1,:], T2[:,k1], Y2[:,k1,:], U1[:,k1,:], Y2hat[:,k1,:])
            t_str = str(T0[-1,k1])
            plt.savefig(figFilePathFormat.format(t_str))
            plt.close(fig)

        figProp = {"prefix": prefix}
        return figProp