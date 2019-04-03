# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:48:10 2019

@author: renxi
"""

import numpy as np

class Statistics:

    def FPmeancov(x, p = None):
        '''This def computes the mean and covariance matrix of a Flexible Probabilities distribution
        Parameters
        x    :[matrix] (i_ x t_end) scenarios
        p    :[vector] ( 1 x t_end) Flexible Probabilities
        Returns
        m    :[vector] (i_ x 1)  mean
        s2   :[matrix] (i_ x i_) covariance matrix'''
        
        i_, t_ = x.shape
       
        if p is None:
            p = np.array([np.ones(t_) / t_])  # equal probabilities as default value
            
        if p.shape[1] == 1:
            p = p.T
        
        m = x.dot(p.T)   # mean

        X_cent = x - np.tile(m, (1, t_))
        s2 = (X_cent*np.tile(p, (i_, 1))).dot(X_cent.T) # covariance matrix
        s2 = (s2 + s2.T)/2 # eliminate numerical error and make covariance symmetric
    
        return m, s2