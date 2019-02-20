""" compares the true efficient frontier, BL prior, BL posterior (general equilibrium + views) """

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

from portfolio.EfficientFrontier import EfficientFrontier

N=10;
MaxVolRets=.4;
MinVolRets=.05;
CorrRets=.0;
#%% # knowledge of the market
CorrRets = (1-CorrRets)*np.eye(N) + CorrRets*np.ones((N,N));
StepVolRets=(MaxVolRets-MinVolRets)/(N-1);
VolRets = np.arange(MinVolRets,MaxVolRets+StepVolRets,StepVolRets)
CovRets= np.diag(VolRets)*CorrRets*np.diag(VolRets);
ExpValRets=2.5*np.dot(CovRets,np.ones((N,1))/N);
NumPortf=15;

# Construct Efficient Frontier
eff = EfficientFrontier(NumPortf, CovRets, ExpValRets);
[E,V,Portfolios] = eff.EfficientFrontier()
Portfolios_True = np.round((100*Portfolios),0)
E_true = E
V_true = V

#Plot
eff.PlotFrontier(Portfolios,'true frontier')
#%% # estimation of the market
R = np.random.multivariate_normal(ExpValRets.reshape((10,)), CovRets, 52*1)
ExpValRets_Hat=np.mean(R,axis=0);
CovRets_Hat=np.cov(R.T);

#%% # prior on the market
CovRets_Prior=CovRets_Hat;
ExpValRets_Prior=2.5*np.dot(CovRets_Hat,np.ones((N,1))/N);

# Construct Efficient Frontier
eff = EfficientFrontier(NumPortf, CovRets_Prior, ExpValRets_Prior);
[E,V,Portfolios] = eff.EfficientFrontier()
E_prior = E
V_prior = V
Portfolios_Prior=np.round(100*Portfolios)

# Plot
eff.PlotFrontier(Portfolios,'prior frontier')
#%% # views on the market
P=np.zeros((2,N));
P[0,0]=1;
P[1,2]=1;
Omega=(3)**2*np.dot(np.dot(P,CovRets_Hat),P.T);
v=np.array([.04, -.02]).reshape((2,1));

# compute posterior BL
middle = np.dot(np.dot(CovRets_Hat,P.T),inv(np.dot(np.dot(P,CovRets_Hat),P.T)+Omega))
Mu_BL=ExpValRets_Prior+np.dot(middle,(v-np.dot(P,ExpValRets_Prior)));
Sigma_BL=CovRets_Hat-np.dot(np.dot(middle,P),CovRets_Hat);

# compute MV efficient frontier
# Construct Efficient Frontier
eff = EfficientFrontier(NumPortf, Sigma_BL, Mu_BL);
[E,V,Portfolios] = eff.EfficientFrontier()
E_BL = E
V_BL = V
Portfolios_BL=np.round(100*Portfolios)
# Plot
eff.PlotFrontier(Portfolios,'BL MV frontier')

#%% Plot Efficient Frontier
plt.figure('Efficient Frontier')
plt.style.use('bmh')
plt.plot(V_true, E_true,'o--', label='true frontier')
plt.plot(V_prior, E_prior,'o-', label='prior frontier')
plt.plot(V_BL, E_BL,'*-', label='BL MV frontier')
plt.legend()
plt.grid(True)
plt.xlabel('Risk(Standard Deviation)')
plt.ylabel('Expected Returns')
plt.show()
