import numpy as np
from math import sqrt

from BlackLitterman import Blacklitterman, display

#// Take the values from the paper
weq = np.array([0.20, 0.20, 0.06, 0.06, 0.06, 0.06, 0.06, 0.30])
C = np.array([[1.00, 0.92, 0.33, 0.26, 0.28, 0.16, 0.29, 0.42],
      [0.92, 1.00, 0.26, 0.22, 0.27, 0.14, 0.25, 0.36],
      [0.33, 0.26, 1.00, 0.41, 0.30, 0.25, 0.58, 0.71],
      [0.26, 0.22, 0.41, 1.00, 0.62, 0.42, 0.54, 0.44],
      [0.28, 0.27, 0.30, 0.62, 1.00, 0.35, 0.48, 0.34],
      [0.16, 0.14, 0.25, 0.42, 0.35, 1.00, 0.40, 0.22],
      [0.29, 0.25, 0.58, 0.54, 0.48, 0.40, 1.00, 0.56],
      [0.42, 0.36, 0.71, 0.44, 0.34, 0.22, 0.56, 1.00]]);
Sigma = np.array([0.054, 0.070, 0.190, 0.244, 0.215, 0.244, 0.208, 0.149]).reshape((8,1));
hPi = np.array([ 0.032, 0.030, 0.046, 0.105, 0.064, 0.105, 0.095, 0.085]).reshape((8,1));
refPi = np.array([0.022, 0.026, 0.096, 0.109, 0.086, 0.078, 0.100, 0.085]).reshape((8,1));
refPi2 = np.array([0.022, 0.026, 0.094, 0.083, 0.063, 0.064, 0.087, 0.092]);
refW2 = np.array([0.20, 0.20, 0.06, 0, 0.015, 0.06, 0.015,  0.45]).reshape((8,1));
miW = np.array([0.23, 0.19, 0.099, 0.043, 0.047, 0.066, 0.054, 0.262]).reshape((8,1));
assets= ['Euro Bonds','US Bonds   ','Canada   ','France  ','Germany  ','Japan    ','UK      ','USA      '];
labels= ['q        ','omega/tau','lambd   '];

#V = np.multiply(np.outer(Sigma,Sigma), C)
V = (Sigma.T * Sigma) * C;
[n,m] = refPi.shape;

#// Risk tolerance of the market from the paper ?????
risk_m = np.dot(np.dot(weq , V) , weq.T);
ret_m = np.dot(weq, refPi);
actual_delta = ret_m / risk_m;

#// Reversed out from refPi, V and weq to match results in paper, not clear
#// where Michaud comes up with this #
delta = 6.82;

#// Reverse optimize equilibrium prior to compare with the paper
pi = np.dot(weq, V) * delta;
  
#// Coefficient of uncertainty in the prior estimate of the mean
#// From the paper
tau = 1.0;

#%%
#// Define view 1
#// US equities will outperform european equities
#// Values for P selected abritrarily
#// Results should match Table 4, Page 21
P1 = np.array([0, 0, 0, -0.40, -0.30, 0, -0.30, 1.00]);
Q1 = np.array([0.05]);
P=P1;
Q=Q1;
#// As specified in the paper, standard error of the estimae is 5#
Omega = np.array([0.05 * 0.05]);
lambd = 0;

res = Blacklitterman(delta, weq, V, tau, P, Q, Omega)
res = res.Meuccibl()

# Display table
d = display('Michaud et al (2013) Table 1',assets,res)
d.disppd(hPi, pi)
Er = res.er
w=res.w
print('View Prior Return\t\t{:.2f}\n'.format( 100*Q[0]));
print('View Prior Std Dev\t\t{:.2f}\n'.format( 100*sqrt(Omega)));
print('Data Return\t\t\t{:.2f}\n'.format(100*np.dot(pi,P.T)));
print('Data Std Deviation\t\t{:.2f}\n'.format(100*sqrt(np.dot(np.dot(P,V),P.T))));
print('View Posterior Return\t\t{:.2f}\n'.format(100*np.dot(Er.T,P.T)[0]));
print('View Effect on Return\t\t{:.2f}\n'.format(100*(np.dot(Er.T,P.T)[0]-np.dot(pi,P.T))));
#%%
Er1 = Er.T;
w1 = w.T;
#// Adjust tau so the numbers look ok (Michaud's ad hoc process because 
#// he uses the Hybrid model
tau = 0.07121798;
P=P1;
Q=Q1;
#// As specified in the paper, standard error of the estimate is 5#
Omega = [0.05 * 0.05];
res = res = Blacklitterman(delta, weq, V, tau, P, Q, Omega)
res = res.Meuccibl()
# Display table
d = display('Michaud et al (2013) Table 2',assets,res)
d.disppd(hPi, pi)
w1=w1.reshape((8,))
Er1=Er1.reshape((8,))
w=w.reshape((8,))
Er=Er.reshape((8,))
print('Return\t\t{:.2f}\t\t\t\t\t{:.2f}\t\t\t\t\t{:.2f}\n'.format(100*np.dot(weq,pi.T), 100*np.dot(w1,Er1.T), 100*np.dot(w.T,Er)));
print('Risk\t\t{:.2f}\t\t\t\t\t{:.2f}\t\t\t\t\t{:.2f}\n'.format( 100*np.sqrt(np.dot(np.dot(weq,V),weq.T)), 100*np.sqrt(np.dot(np.dot(w1,V),w1.T)), 100*np.sqrt(np.dot(np.dot(w.T,V),w))));

#%%
Erm = Er;
#// Resample with replacement/Bootstrap to bound tau, the dataset used
#// in Michaud, et all (2013) has 216 data pointa. These numbers seem low
#// for numbers of Monte Carlo simulations in an activity founded on 
#// the Central Limit theory.  Here we use 1000 
Z = np.zeros((10000,n))
for i in range(10000):
    Y =  np.random.multivariate_normal(pi, V, 216);
    Z[i,:] = np.mean(Y,axis=0);
t4=np.zeros((n,5));
from scipy.stats import norm
def norminv(p,mean,std):
    from scipy.stats import norm
    rv = norm(mean, std).ppf(p)
    return rv
mean = np.mean(Z,axis=0)
std = np.std(Z,axis=0)
t4 = norm(mean,std).ppf([[0.05],[0.25],[0.5],[0.75],[0.95]]).T
t4 = 100* t4;

#%%
#// Back out tau too
sd = np.sqrt(np.diag(V));
t = np.std(Z,axis=0);
tx = (sd * sd) /(t * t);
print('Michaud et al (2013) Table 4\n');
print('Percentiles    5%   25%\t BL Means   75%    95%\t 1/Tau\n');
import pandas as pd
result = pd.DataFrame(t4, columns=['5%','25%','BL Means','75%','95%'])
result['Asset Name'] = assets
result = result.set_index('Asset Name')
result['1/Tau'] = tx
print (result.round(2).to_string())

