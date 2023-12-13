import copy
import math
import torch
import torch.nn as nn
import numpy as np
import cupy as cp
try:
    from Tools import phi,pVar
except:
    from .Tools import phi, pVar
    
class rnnLU(nn.Module):
    """
    rnn model with linear-nonlinear dynamics
    """
    def __init__(self,g=1, N=500,device='cpu'):
        super(rnnLU, self).__init__()
        self.g = g  # initial weight scale factor
        self.P0 = 1
        self.device=device
        self.N=N

    def set_input(self, input):
        self.input = input

    def set_target(self,target=None):
        """
        Args:
            mode: 'neural' one target for each unit
                  'output' output targets of M channel
            target: ndarray of N (neural target) or M (output target) dim.
        """
        self.target = target
        self.target_dim = self.target.shape[0]
        self.n_step = self.target.shape[1]

    def _begin_network_state(self):
        self.pred = cp.zeros((self.target_dim, self.n_step))
        x0=cp.zeros(self.N)
        return x0
    
    def set_params(self,J,W=None,Wfb=None):
        self.J = J
        if W is not None:
            self.W = W
            self.Wfb = Wfb

    def initialize_params(self):
        # Recurrent weights
        self.J = self.g * cp.random.randn(self.N, self.N) / math.sqrt(self.N)
        self.W = cp.zeros((self.target_dim, self.N))
        self.Wfb = self.g * cp.random.randn(self.N, self.target_dim) / math.sqrt(self.target_dim)
    
    def _step(self,x,input,output=None):
        #forward pass
        x = cp.dot(self.J, phi(x, 'tanh'))  # rec+input+feedback
        x = x + input
        x= x+ cp.dot(self.Wfb, output)
        return x

    def _save(self,n,pred_n):
        #save the n+1 data to memory
        self.pred[:, n] = pred_n

    def run(self,J=None):
        x0=self._begin_network_state()
        for i in range(self.n_step):
            # do the integration
            if i == 0:
                x = x0
            else:
                x = self._step(x,self.input[:, i],output)
            output=cp.dot(self.W, phi(x, 'tanh'))
            self._save(i,output)
        return self.pred
        
    def train(self,optimizer,numEpoch=1):
        pVar_learningCurve = []
        mse_learningCurve = []
        best_param=None
        for nEpoch in range(numEpoch):
            x0=self._begin_network_state()
            for i in range(self.n_step):
                # do the integration
                if i == 0:
                    x = x0
                else:
                    x = self._step(x,self.input[:, i],output)
                output=cp.dot(self.W, phi(x, 'tanh'))
                self._save(i,output)
                if i>0:
                    target = self.target[:,i]
                    error=output-target
                    u=phi(x,'tanh')
                    oldW=copy.deepcopy(self.W)
                    self.W=optimizer.update(self.W,u,error)

            pVar_learningCurve.append(pVar(self.target, self.pred))
            mse_learningCurve.append(cp.mean(cp.square(self.target-self.pred)))
            if mse_learningCurve[-1]==cp.min(cp.array(mse_learningCurve)):
                best_param=copy.deepcopy(self.W)
            
        return pVar_learningCurve,mse_learningCurve,self.pred,best_param
        
        
        
