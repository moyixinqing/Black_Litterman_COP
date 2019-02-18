# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:20:57 2019

@author: renxi
"""
import os
import os.path as path
import sys

sys.path.append(path.join(path.dirname(path.dirname(path.abspath('.'))), 'Meucci_2006_CopulaOpinionPooling'))

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from CONFIG import GLOBAL_DB, TEMPORARY_DB
from PortfolioCVaR import PortfolioCVaR
#%% Step 1 Input Data 
# load matrix data of annualized linear stock returns
try:
    db = loadmat(os.path.join(GLOBAL_DB, 'LinRetMat'), squeeze_me=True)
except FileNotFoundError:
    db = loadmat(os.path.join(TEMPORARY_DB, 'LinRetMat'), squeeze_me=True)

data = db['data']

#data = MPost[:30,:]

#%% Step 2 Mean_CVaR Portfolio Optimization
NPort = 49
quantile = 0.97

port = PortfolioCVaR(data, NPort, quantile)
[weights,Rstar,CVaR] = port.CVaROpt_nonNormal()
#Plot
port.plot(CVaR,Rstar,'Mean-CVaR Frontier')
p_=np.argmin(CVaR)
pi_=np.argmax(CVaR[p_:]>0)+p_+1
port.PlotCVaRFrontier(weights[:,pi_:].T,CVaR[pi_:],'Weights of Mean-CVaR Efficient Portfolio')
