'''
Created on 2020/07/16

@author: ukai
'''


import numpy as np
from data_generator_abstract_singleton import DataGeneratorAbstractSingleton


class DataGeneratorSingleton(DataGeneratorAbstractSingleton):
    '''
    classdocs
    '''
            
    def __init__(self, Nhidden, Ntrain = 2**12, T0 = 2**1, T1 = 2**7, 
        Ny = None, Nu = None, Nw = None, prob_step = 1/2**4, action_distribution = "normal", amp_dv = 1.0, seed = 0):
        
        print("""\
        
        ========================
        DataGeneratorSingleton's constructor was called with the following parameters:
        Nhidden: {0}
        Ntrain: {1}
        T0: {2}
        T1: {3}
        Ny: {4}
        Nu: {5}
        Nw: {6}
        action_distribution: {7}
        amp_dv: {8}
        seed: {9}
        ========================        
        
        """.format(Nhidden, Ntrain, T0, T1, Ny, Nu, Nw, action_distribution, amp_dv, seed))
        
        rstate = np.random.RandomState(seed)
        
        Ny = Nhidden if Ny is None else Ny
        Nu = Nhidden if Nu is None else Nu
        Nw = 0 if Nw is None else Nw
        Nhalf = Nhidden//2

# pole/continuous time system = -alpha + 1j * beta * 2 * pi
        alpha = 1/np.exp(np.log(T0) + rstate.rand(Nhalf) * (np.log(T1) - np.log(T0)))
        beta  = 1/np.exp(np.log(T0) + rstate.rand(Nhalf) * (np.log(T1) - np.log(T0)))

        Diag = np.diag(np.concatenate([np.exp(-alpha + 1j * beta * np.pi), np.exp(-alpha - 1j * beta * np.pi)], axis=0))
        Vr = rstate.randn(Nhidden, Nhalf) 
        Vi = rstate.randn(Nhidden, Nhalf) 
        V = np.concatenate([Vr + 1j * Vi, Vr - 1j * Vi], axis=1)

        A = np.real(np.dot( np.dot(V, Diag), np.linalg.inv(V)))
        multiplier = np.concatenate((np.sqrt(1-np.exp(-2*alpha)), np.sqrt(1-np.exp(-2*alpha))))
        B = multiplier.reshape((-1,1)) *  rstate.randn(Nhidden, Nu + Nw)
        C = rstate.randn(Ny, Nhidden)

        x = rstate.randn(Nhidden)
        X = []

        if action_distribution == "step":
            #--------------#
            U = []
            Ev = []
            u = np.zeros(Nu) # (Nu)
            for k1 in range(Ntrain):
                r = rstate.rand(Nu) < prob_step
                u[r] = 1 - u[r]
                U.append(u.copy())
                Ev.append(np.any(r))
            U = np.stack(U, axis=0) # (*, Nu)
            Ev = np.array(Ev)
            #--------------#
            
        if action_distribution == "normal":
            U = rstate.randn(Ntrain, Nu)
            Ev = np.ones(Ntrain) # (Ntrain)
            
        W = rstate.randn(Ntrain, Nw) * amp_dv
        UW = np.concatenate((U, W), axis=-1)        
        for k1 in range(Ntrain):
            uw = UW[k1,:]
            x = np.dot(A, x) + np.dot(B, uw)
            X.append(x)
        X = np.stack(X, axis=0)
        Y = np.dot(X, C.T)
        Y = (Y - np.mean(Y, axis=0))/np.std(Y, axis=0)

        self.A = A
        self.B = B
        self.C = C
        self.U = U.astype(np.float32)
        self.Ev = Ev
        self.W = W.astype(np.float32) # W should not be used.
        self.Y = Y.astype(np.float32)
        self.Ntrain = Ntrain
        self.Nhidden = Nhidden
        self.Nu = Nu
        self.Ny = Ny
