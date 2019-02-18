# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:59:24 2019

@author: renxi
"""
import numpy as np
from copulas.multivariate.gaussian import GaussianMultivariate

class Estimation:

    def cov_2_corr(s2):
        """
        Inputs
        ----------
            s2 : array, shape (n_, n_)
        Outputs
        -------
            c2 : array, shape (n_, n_)
            s_vol : array, shape (n_,)
        """
        s_vol = np.sqrt(np.diag(s2))
        c2 = (s2 / s_vol).T / s_vol
        return c2, s_vol

    def fit_Gcopula(data):
        gc = GaussianMultivariate()
        gc.fit(data)
        print(gc)
        
