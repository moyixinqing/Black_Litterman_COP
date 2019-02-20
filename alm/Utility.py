# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:42:36 2019

@author: renxi
"""

from numpy.linalg import svd
from copulas.multivariate.gaussian import GaussianMultivariate
import pandas as pd
from scipy.stats import skew, kurtosis

from alm.Estimation import Estimation

class Utility:
    """frequrntly used functions"""
    def nullspace(a, rtol=1e-5):
        """

        Parameters
        ----------
        a :
            
        rtol :
            (Default value = 1e-5)

        Returns
        -------

        
        """
        u, s, v = svd(a)
        rank = (s > rtol*s[0]).sum()
        return rank, v[rank:].T.copy()

    def disp_stat(M, Name):        
        """

        Parameters
        ----------
        M :
            
        Name :
            

        Returns
        -------

        
        """
        NInput = len(M)        
        for i in range(NInput):
            data = M[i]
            name = Name[i]    
            mu = data.mean(axis=0)
            sigma = data.std(axis=0)
            sk = skew(data)
            ku = kurtosis(data)            
            dist = pd.DataFrame()
            dist ['mu'] = mu
            dist ['sigma'] = sigma
            dist ['sk'] = sk
            dist ['ku'] = ku
            print('{} market statistics _ Gaussian Copula'.format(name))
            print(dist.round(2).T.to_string())
            gc = GaussianMultivariate()
            gc_prior_res = gc.fit(data)
            X_corr,_ = Estimation.cov_2_corr(gc.covariance)
            print('{} correlation'.format(name))
            print(X_corr)
            print('\n')