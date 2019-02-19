# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 11:15:02 2018

@author: renxi
"""

import numpy as np
from BlackLitterman import Blacklitterman, display
 
#%%
# Take the values from He & Litterman, 1999.
weq = np.array([0.016,0.022,0.052,0.055,0.116,0.124,0.615])
C = np.array([[ 1.000, 0.488, 0.478, 0.515, 0.439, 0.512, 0.491],
      [0.488, 1.000, 0.664, 0.655, 0.310, 0.608, 0.779],
      [0.478, 0.664, 1.000, 0.861, 0.355, 0.783, 0.668],
      [0.515, 0.655, 0.861, 1.000, 0.354, 0.777, 0.653],
      [0.439, 0.310, 0.355, 0.354, 1.000, 0.405, 0.306],
      [0.512, 0.608, 0.783, 0.777, 0.405, 1.000, 0.652],
      [0.491, 0.779, 0.668, 0.653, 0.306, 0.652, 1.000]])
Sigma = np.array([0.160, 0.203, 0.248, 0.271, 0.210, 0.200, 0.187])
refPi = np.array([0.039, 0.069, 0.084, 0.090, 0.043, 0.068, 0.076])
assets= ['Australia','Canada   ','France   ','Germany  ','Japan    ','UK       ','USA      ']

# Equilibrium covariance matrix
V = np.multiply(np.outer(Sigma,Sigma), C)
#V = Sigma.T*Sigma*C

# Risk aversion of the market 
delta = 2.5

# Coefficient of uncertainty in the prior estimate of the mean
# from footnote (8) on page 11
tau = 0.05
tauV = tau * V

#%%
# Define view 1
# Germany will outperform the other European markets by 5%
# Market cap weight the P matrix
# Results should match Table 4, Page 21
P1 = np.array([0, 0, -.295, 1.00, 0, -.705, 0 ])
Q1 = np.array([0.05])
P=np.array([P1])
Q=np.array([Q1]);
Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])
res = Blacklitterman(delta, weq, V, tau, P, Q, Omega)
res = res.blacklitterman()
d = display('Table 4: View 1 (He & Litterman 1999)',assets,res)
d.disp()
#%%
# Define view 2
# Canadian Equities will outperform US equities by 3%
# Market cap weight the P matrix
# Results should match Table 5, Page 22
P2 = np.array([0, 1.0, 0, 0, 0, 0, -1.0 ])
Q2 = np.array([0.03])
P=np.array([P1,P2])
Q=np.array([Q1,Q2]);
Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])
res = Blacklitterman(delta, weq, V, tau, P, Q, Omega)
res = res.blacklitterman()
d = display('Table 5: View 1 & 2 (He & Litterman 1999)', assets, res)
d.disp()
#%%
# Define view 3
# Canadian Equities will outperform US equities by 3%
# Market cap weight the P matrix
# Results should match Table 5, Page 22
P2 = np.array([0, 1.0, 0, 0, 0, 0, -1.0 ])
Q2 = np.array([0.04])
P=np.array([P1,P2])
Q=np.array([Q1,Q2]);
Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])
Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])
res = Blacklitterman(delta, weq, V, tau, P, Q, Omega)
res = res.blacklitterman()
d = display('Table 6: View 1 + 2, View 2 more bullish (He & Litterman 1999)', assets, res)
d.disp()
#%%
# Define view 4
#// Decrease confidence in view 1
#// Double the variance of view 1
#// Results should match Table 7, Page 24
P2 = np.array([0, 1.0, 0, 0, 0, 0, -1.0 ])
Q2 = np.array([0.04])
P=np.array([P1,P2])
Q=np.array([Q1,Q2]);
Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])
Omega[0,0]=2*Omega[0,0];
res = Blacklitterman(delta, weq, V, tau, P, Q, Omega)
res = res.blacklitterman()
d = display('Table 7: View 1 & 2, View 1 less confident (He & Litterman 1999)', assets, res)
d.disp()
#%%
# Define view 5
#// Add a new view which matches the results from view 1 & 2 to show it 
#// doesn't change things
P3 = np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0 ]);
Q3 = np.array([0.0412]);
P = np.array([P1, P2, P3]);
Q = np.array([Q1, Q2, Q3]);
Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])
Omega[0,0]=2*Omega[0,0];
res = Blacklitterman(delta, weq, V, tau, P, Q, Omega)
res = res.blacklitterman()
d = display('Table 8: View 1, 2 & 3, View 1 less confident (He & Litterman 1999)', assets, res)
d.disp()
