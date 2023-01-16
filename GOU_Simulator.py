# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:11:29 2022

@author: sebja
"""

import numpy as np

class Simulator:
    
    def __init__(self, T=0.25, NdT=89, S0=10, kappa=2, theta=np.log(10), sigma=0.4):
        
        self.T = T
        self.NdT = NdT
        self.t = np.linspace(0,T,self.NdT+1)
        self.dt = self.t[1]-self.t[0]
        
        self.S0 = S0
        
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        
        self.theta_X = self.theta - 0.5*self.sigma**2/self.kappa
        
        
    def Sim(self, batch_size = 1024):
        
        X = np.zeros((batch_size, self.NdT+1))
        X[:,0] = np.log(self.S0)
        
        for n in range(self.NdT):
            
            X[:,n+1] = self.theta_X + np.exp(-self.kappa*self.dt) * (X[:,n] - self.theta_X)
            X[:,n+1] += self.sigma*np.sqrt(self.dt)* np.random.randn(batch_size)
            
        S = np.exp(X)
        
        return S
        
        