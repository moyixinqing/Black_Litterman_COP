import numpy as np
from scipy import linalg

class Blacklitterman(object):
    """The base class for Black Litterman.
    This function performs the Black-Litterman blending of the prior
    and the views into a new posterior estimate of the returns as
    described in the paper by He and Litterman.
   
    ----------
     Inputs
       delta  - Risk tolerance from the equilibrium portfolio
       weq    - Weights of the assets in the equilibrium portfolio
       sigma  - Prior covariance matrix
       tau    - Coefficiet of uncertainty in the prior estimate of the mean (pi)
       P      - Pick matrix for the view(s)
       Q      - Vector of view returns
       Omega  - Matrix of variance of the views (diagonal)
    Outputs
       Er     - Posterior estimate of the mean returns
       w      - Unconstrained weights computed given the Posterior estimates
            of the mean and covariance of returns.
       lambda - A measure of the impact of each view on the posterior estimates.
    """
    def __init__(self,delta, weq, sigma, tau, P, Q, Omega):
        self.delta = delta
        self.weq = weq
        self.sigma = sigma
        self.tau= tau 
        self.P = P
        self.Q = Q
        self.Omega = Omega
        
    def blacklitterman(self):

      # Reverse optimize and back out the equilibrium returns
      # This is formula (12) page 6.
      pi = self.weq.dot(self.sigma * self.delta)
      #print(pi)
      # We use tau * sigma many places so just compute it once
      ts = self.tau * self.sigma
      # Compute posterior estimate of the mean
      # This is a simplified version of formula (8) on page 4.
      middle = linalg.inv(np.dot(np.dot(self.P,ts),self.P.T) + self.Omega)
      #print(middle)
      #print(Q-np.expand_dims(np.dot(P,pi.T),axis=1))
      er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(ts,self.P.T),middle),(self.Q - np.expand_dims(np.dot(self.P,pi.T),axis=1)))
      # Compute posterior estimate of the uncertainty in the mean
      # This is a simplified and combined version of formulas (9) and (15)
      posteriorSigma = self.sigma + ts - ts.dot(self.P.T).dot(middle).dot(self.P).dot(ts)
      #print(posteriorSigma)
      # Compute posterior weights based on uncertainty in mean
      w = er.T.dot(linalg.inv(self.delta * posteriorSigma)).T
      # Compute lambda value
      # We solve for lambda from formula (17) page 7, rather than formula (18)
      # just because it is less to type, and we've already computed w*.
      lmbda = np.dot(linalg.pinv(self.P).T,(w.T * (1 + self.tau) - self.weq).T)
      
      change=(w[:,0]-self.weq/(1+self.tau));
      for i in range(len(change)):
            if abs(change [i]) < 1e-10:
                change[i] = 0;
                
      return [er, w, lmbda, change]
    

    def altblacklitterman(self):
        """
        alternative Black-Litterman 
        """
        # Reverse optimize and back out the equilibrium returns
        # This is formula (12) page 6.
        pi = self.weq.dot(self.sigma * self.delta)
        # We use tau * sigma many places so just compute it once
        ts = self.tau * self.sigma
        # Compute posterior estimate of the mean
        # This is a simplified version of formula (8) on page 4.
        middle = linalg.inv(np.dot(np.dot(self.P,ts),self.P.T) + self.Omega)
        er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(ts,self.P.T),middle),(self.Q - np.expand_dims(np.dot(self.P,pi.T),axis=1)))
        # Compute posterior estimate of the uncertainty in the mean
        # This is a simplified and combined version of formulas (9) and (15)
        # Compute posterior weights based on uncertainty in mean
        w = er.T.dot(linalg.inv(self.delta * self.sigma)).T
        # Compute lambda value
        # We solve for lambda from formula (17) page 7, rather than formula (18)
        # just because it is less to type, and we've already computed w*.
        lmbda = np.dot(linalg.pinv(self.P).T,(w.T * (1 + self.tau) - self.weq).T)
       
        return [er, w, lmbda]
    
    def bl_omega(self, conf, P, Sigma):
        """ idz_omega
        #   This function computes the Black-Litterman parameters Omega from
        #   an Idzorek confidence.
        # Inputs
        #   conf   - Idzorek confidence specified as a decimal (50% as 0.50)
        #   P      - Pick matrix for the view
        #   Sigma  - Prior covariance matrix
        # Outputs
        #   omega  - Black-Litterman uncertainty/confidence parameter
        """
        alpha = (1 - conf) / conf
        omega = alpha * np.dot(np.dot(P,Sigma),P.T)
        return omega

class display(Blacklitterman):
    
    def __init__(self, tau, P, Q, Omega, title, assets, res):
        self.tau = tau
        self.P = P
        self.Q = Q
        self.Omega = Omega
        
        self.title = title
        self.assets = assets
        self.res = res
        
    def display(self):
      """
       Function to display the results of a black-litterman shrinkage
       Inputs
           title    - Displayed at top of output
           assets    - List of self.assets
           res        - List of results structures from the bl function
      """
      er = self.res[0]
      w = self.res[1]
      lmbda = self.res[2]
      print('\n' + self.title)
      line = 'Country\t\t'
      for p in range(len(self.P)):
        line = line + 'P' + str(p) + '\t'
      line = line + 'mu\tw*'
      print(line)
      i = 0;
      for x in self.assets:
        line = '{0}\t'.format(x)
        for j in range(len(self.P.T[i])):
            line = line + '{0:.1f}\t'.format(100*self.P.T[i][j])
    
        line = line + '{0:.3f}\t{1:.3f}'.format(100*er[i][0],100*w[i][0])
        print(line)
        i = i + 1
      line = 'q\t\t'
      i = 0
      for q in self.Q:
        line = line + '{0:.2f}\t'.format(100*q[0])
        i = i + 1
      print(line)
      line = 'omega/tau\t'
      i = 0
      for o in self.Omega:
        line = line + '{0:.5f}\t'.format(o[i]/self.tau)
        i = i + 1
      print(line)
      line = 'lambda\t\t'
      i = 0
      for l in lmbda:
        line = line + '{0:.5f}\t'.format(l[0])
        i = i + 1
      print(line)    
    
    def disp (self):
        
        import pandas as pd
        line = ['Country']
        for p in range(len(self.P)):
            line = line + ['P{}'.format(p)]      
        columns = line + ['mu','w*','w*-weq/(1+tau)']
        self.result = pd.DataFrame(columns= columns)
        self.result ['Country'] = np.array(self.assets)
        self.result = self.result.set_index('Country')
        for p in range(len(self.P)):
            self.result ['P{}'.format(p)] = np.array(self.P[p])*100
        self.result ['mu'] = np.array(self.res[-4])*100
        self.result ['w*'] = np.array(self.res[-3])*100
        self.result ['w*-weq/(1+tau)'] = np.array(self.res[-1])*100
        print(self.title)
        print('------------------------------------------------------------------')
        print(self.result.round(1))
        print('------------------------------------------------------------------')
        line = 'q\t\t'
        i = 0
        for q in self.Q:
            line = line + '{0:.2f}\t'.format(100*q[0])
            i = i + 1
        print(line)
        line = 'omega/tau\t'
        i = 0
        for o in self.Omega:
            line = line + '{0:.3f}\t'.format(o[i]/self.tau)
            i = i + 1
        print(line)
        line = 'lambda\t\t'
        i = 0
        lmbda = self.res[-2]
        for l in lmbda:
            line = line + '{0:.3f}\t'.format(l[0])
            i = i + 1
        print(line)
        print('------------------------------------------------------------------\n')