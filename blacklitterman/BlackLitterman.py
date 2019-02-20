import numpy as np

from scipy import linalg
from collections import namedtuple

class Blacklitterman(object):
    """The base class for Black Litterman."""

    def __init__(self, delta, weq, sigma, tau, P, Q, Omega):
      
        self.delta = delta
        self.weq = weq
        self.sigma = sigma
        self.tau= tau
        self.P = P
        self.Q = Q
        self.Omega = Omega

    def blacklitterman(self):
        """This function performs the Black-Litterman blending of the prior
        and the views into a new posterior estimate of the returns as
        described in the paper by He and Litterman.

        Parameters
        ----------
        delta :
            Risk tolerance from the equilibrium portfolio
        weq :
            Weights of the assets in the equilibrium portfolio
        sigma :
            Prior covariance matrix
        tau :
            Coefficiet of uncertainty in the prior estimate of the mean
        P :
            Pick matrix for the view
        Q :
            Vector of view returns
        Omega :
            Matrix of variance of the views

        Returns
        -------

        
        """
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
      
        res = namedtuple('res', ['er', 'w', 'lmbda', 'change', 'delta', 'weq','V', 'tau', 'P', 'Q', 'Omega'])
        res.er = er
        res.w = w
        res.lmbda = lmbda
        res.change = change
        res.delta = self.delta
        res.V = self.sigma
        res.weq = self.weq
        res.tau = self.tau
        res.P = self.P
        res.Q = self.Q
        res.Omega = self.Omega
        return res     
    
    def altblacklitterman(self):
        """alternative Black-Litterman"""
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
        
        res = namedtuple('res', ['er', 'w', 'lmbda', 'delta', 'weq','V', 'tau', 'P', 'Q', 'Omega'])
        res.er = er
        res.w = w
        res.lmbda =lmbda
        res.delta = self.delta
        res.V = self.sigma
        res.weq = self.weq
        res.tau = self.tau
        res.P = self.P
        res.Q = self.Q
        res.Omega = self.Omega
        return res        
                
    def bl_omega(self, conf, P, Sigma):
        """Idzorek_omega
        This function computes the Black-Litterman parameters Omega from
           an Idzorek confidence.

        Parameters
        ----------
        conf :
            Idzorek confidence specified as a decimal
        P :
            Pick matrix for the view
        Sigma :
            Prior covariance matrix

        Returns
        -------

        
        """
        alpha = (1 - conf) / conf
        omega = alpha * np.dot(np.dot(P,Sigma),P.T)
        return omega

    def Meuccibl(self):
        """This function performs the Black-Litterman blending of the prior
        and the views into a new posterior estimate of the returns as
        described in the various papers by Meucci.

        Parameters
        ----------
        delta :
            Risk tolerance from the equilibrium portfolio
        weq :
            Weights of the assets in the equilibrium portfolio
        sigma :
            Prior covariance matrix
        tau :
            Coefficiet of uncertainty in the prior estimate of the mean
        P :
            Pick matrix for the view
        Q :
            Vector of view returns
        Omega :
            Matrix of variance of the views

        Returns
        -------

        
        """

        #// Reverse optimize and back out the equilibrium returns
        #// This is formula (12) page 6.
        pi = np.dot(self.weq, self.sigma) * self.delta;

        #// We use tau * sigma many places so just compute it once
        ts = self.tau * self.sigma;
        #// Compute posterior estimate of the mean
        #// This is a simplified version of formula (8) on page 4.
        pi = np.expand_dims(pi ,axis=0)
        P = np.expand_dims(self.P.T,axis=0)
        er = pi.T + np.dot(ts , P.T) * linalg.inv(np.dot(np.dot(P, ts),P.T) + self.Omega) * (self.Q - np.dot(P , pi.T));

        #// Compute posterior weights based on uncertainty in mean
        w = (np.dot(er.T , linalg.inv(self.delta * self.sigma))).T;

        res = namedtuple('res', ['er', 'w', 'delta', 'weq','V', 'tau', 'P', 'Q', 'Omega'])
        res.er = er
        res.w = w
        res.delta = self.delta
        res.V = self.sigma
        res.weq = self.weq
        res.tau = self.tau
        res.P = self.P
        res.Q = self.Q
        res.Omega = self.Omega
        return res
