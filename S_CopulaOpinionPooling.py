# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 14:13:28 2019

@author: renxi

' Two methods for Market prior distribution
'              1. Gaussian Copula fit and simulation
'              2. tCopula fit and simulation
' Three methods for View distribution
'              1. Gaussian View
'              2. Uniform View
'              3. Uniform View with Market Flexible Probabilities
' Furute for View distribution
'              4. dcc incoporate exponential time decay flexible probabilities
'
"""

import numpy as np
from numpy import ones
import pandas as pd

from copulas.multivariate.gaussian import GaussianMultivariate

from alm.Utility import Utility
from alm.Estimation import Estimation
from blacklitterman.COP import COP
from portfolio.PortfolioCVaR import PortfolioCVaR


# Input parameters

n_ = 6  # number of stocks
t_first = '2012-01-01'  # starting date
t_last = '2019-01-01'  # ending date
k_ = 3  # number of factors
nu = 4.  # degrees of freedom
tau_hl = 120  # prior half life
i_1 = 2  # index of first quasi-invariant shown in plot
i_2 = 5  # index of second quasi-invariant shown in plot

view =  'Uniform' # view disribution

# Load Data

path = 'C:/Users/renxi/Documents/GitHub/Black_Litterman_Asset_Allocation/databases/global-databases/401k/'
data = pd.read_csv(path + '401k_Price.csv', index_col=0)
# set timestamps
data = data.set_index(pd.to_datetime(data.index))
# select data within the date range
data = data.loc[(data.index >= t_first) & (data.index <= t_last)]
# remove the stocks with missing values
data = data.dropna(axis=1, how='any')
data = np.array(data.iloc[:, :n_])
# compounded return
data = np.diff(np.log(data), axis=0)

# Copula Opinion Pooling Input

P_mat = np.array([[0, -1, 0, 0, 1, 0],
                  [0, -0.5, 1, -0.5, 0, 0]])  # views
mu_v = np.array([0.0006, 0.0007])         # normal view mean
sigma_v = np.array([0.05, 0.075])         # normal view sigma
range_v = np.array([[0, 0.01],
                    [0, 0.02]])        # uniform view range
k_ = P_mat.shape[0]                       # number of views
Conf_full = ones((k_, 1)) - 1e-6          # full confidence levels
Conf = ones((k_, 1)) * 0.25                 # half confidence levels

# Market Information fit and simulation

# Market Prior from Gaussian Copula Simulation
gc = GaussianMultivariate()
gc.fit(data)
print(gc)
M = gc.sample(1000)
Utility.disp_stat([M], ['Prior'])
MPrior = M.values

name = 'True'
print('Mean_{}'.format(name))
mean = np.expand_dims(data.mean(axis=0), axis=0).T
print(data.mean(axis=0).round(2))
print('-------------------------------------------------\n')
df_cov = pd.DataFrame(data=gc.covariance)
gc_corr = Estimation.cov_2_corr(gc.covariance)[0]
gc_sigvec = Estimation.cov_2_corr(gc.covariance)[1]
df_corr = pd.DataFrame(data=gc_corr)
print('Correlation_{}'.format(name))
print(df_corr.round(4))
print('-------------------------------------------------\n')
df_sigvec = pd.DataFrame(data=gc_sigvec)
print('Sigvec_{}'.format(name))
print(df_sigvec.T.round(4))

# Market Prior from Student T Copula Simulation
# MPrior = Z.T   #generated from Total Return Generator

# Copula Opinion View  
if view == 'Gaussian':
    # Gaussian View    
    post = COP(MPrior, Conf, P_mat)
    MPost = post.Gassian_View(mu_v, sigma_v)
    print('Gaussian View')
    Utility.disp_stat([MPrior, MPost], ['Prior', 'Post'])
elif view == 'Uniform':    
    # Uniform View    
    post = COP(MPrior, Conf, P_mat)
    MPost = post.Uniform_View(range_v)
    print('Uniform View')
    Utility.disp_stat([MPrior, MPost], ['Prior', 'Post'])
    post.Plot_cdf(range_v, 'Cdf of the steepening view')
# %%
# Construct Portfolio

# Mean_CVaR Portfolio Optimization

NPort = 50
quantile = 0.97
port = PortfolioCVaR(data, NPort, quantile)
[weights, Rstar, CVaR] = port.CVaROpt_nonNormal()

# Plot
port.plot(CVaR, Rstar, 'Mean-CVaR Frontier')
p_ = np.argmin(CVaR)
pi_ = np.argmax(CVaR[p_:] > 0) + p_ + 1
port.PlotCVaRFrontier(weights[:, pi_:].T, CVaR[pi_:],
                      'Weights of Mean-CVaR Efficient Portfolio')
