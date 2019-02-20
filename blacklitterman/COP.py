# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 21:26:19 2019

@author: renxi
"""
from numpy import zeros, r_, c_
from numpy.linalg import inv
import numpy as np
from scipy.stats import norm
from scipy import interpolate
from scipy.stats import uniform
import matplotlib.pyplot as plt

from alm.Utility import Utility

class COP:
    """The base class for Copula Opinion Pooling."""
    
    def __init__(self, MPrior, Conf, P_mat):
        self.MPrior = MPrior
        self.Conf = Conf 
        self.P_mat = P_mat

    def Gassian_View(self, mu_v, sigma_v):
        """

        Parameters
        ----------
        mu_v :
            
        sigma_v :
            

        Returns
        -------

        
        """

        NSim,_ = self.MPrior.shape; # get NSim
        NViews = self.P_mat.shape[0]; # get NViews
        
        #P_bar=[P_mat; null(P_mat).T]; # compute P_bar
        P_ort = Utility.nullspace(self.P_mat)[1].T
        P_bar = r_[self.P_mat,  P_ort]
    
        V=self.MPrior@P_bar.T;  # transform input
        
        W=np.sort(V[:,:NViews],axis=0) # for empirical copula
        Cin=np.array(V[:,:NViews].argsort(axis=0),dtype=np.float64)+1 # for empirical copula
        Grid = np.arange(NSim)+1.0;
        
        C = zeros(Cin.shape)
        for k in range(NViews):
            f = interpolate.interp1d(Cin[:,k],Grid);
            C[:,k] = f(Grid)/(NSim+1);
            
        F = zeros((NSim,NViews)); 
        F_hat = zeros((NSim,NViews));
        F_tilda = zeros((NSim,NViews));
        V_tilda = zeros((NSim,NViews));
        
        for k in range(NViews): # determine the posterior margianl per view
            F[:,k]= Grid.T/(NSim+1);
            # Gaussian view
            F_hat[:,k]=norm.cdf(W[:,k],mu_v[k],sigma_v[k]);
            F_tilda[:,k]=(1-self.Conf[k])*F[:,k] \
                + self.Conf[k]*F_hat[:,k]; #  weighted distribution
            # joint postorior
            f = interpolate.interp1d(F_tilda[:,k], W[:,k],fill_value='extrapolate');
            V_tilda[:,k] = f(C[:,k]);
        
        V_tilda=c_[V_tilda, V[:,NViews:]]; # joint posterior distribution
        MPost=V_tilda@inv(P_bar.T); # new distribution incl. views
        
        return MPost 

    def Uniform_View(self, range_v):
        """

        Parameters
        ----------
        range_v :
            

        Returns
        -------

        
        """
        
        NSim,_ = self.MPrior.shape; # get NSim
        NViews = self.P_mat.shape[0]; # get NViews
        
        P_ort = Utility.nullspace(self.P_mat)[1].T # compute P_bar
        P_bar = r_[self.P_mat,  P_ort]     # compute P_bar
    
        V=self.MPrior@P_bar.T;  # transform input
        
        W=np.sort(V[:,:NViews],axis=0)  # for empirical copula
        Cin=np.array(V[:,:NViews].argsort(axis=0),dtype=np.float64)+1 # for empirical copula
        Grid = np.arange(NSim)+1.0;
        
        C = zeros(Cin.shape)
        for k in range(NViews):
            f = interpolate.interp1d(Cin[:,k],Grid);
            C[:,k] = f(Grid)/(NSim+1);
            
        F = zeros((NSim,NViews)); 
        F_hat = zeros((NSim,NViews));
        F_tilda = zeros((NSim,NViews));
        V_tilda = zeros((NSim,NViews));
        
        for k in range(NViews): # determine the posterior margianl per view
            F[:,k]= Grid.T/(NSim+1);
            # Uniform view
            F_hat[:,k]=uniform.cdf(W[:,k], range_v[k,0], range_v[k,1])
            F_tilda[:,k]=(1-self.Conf[k])*F[:,k] + self.Conf[k]*F_hat[:,k]; #  weighted distribution
            # joint postorior
            f = interpolate.interp1d(F_tilda[:,k], W[:,k],fill_value='extrapolate');
            V_tilda[:,k] = f(C[:,k]);
        
        V_tilda=c_[V_tilda, V[:,NViews:]]; # joint posterior distribution
        MPost=V_tilda@inv(P_bar.T); # new distribution incl. views
        return MPost 
        
    def Plot_cdf(self, range_v, name):
        """

        Parameters
        ----------
        range_v :
            
        name :
            

        Returns
        -------

        
        """
        # Cdf Input
        NSim,_ = self.MPrior.shape; # get NSim
        NViews = self.P_mat.shape[0]; # get NViews
        
        P_ort = Utility.nullspace(self.P_mat)[1].T # compute P_bar
        P_bar = r_[self.P_mat,  P_ort]     # compute P_bar
        
        V = self.MPrior@P_bar.T;  # transform input
        
        W = np.sort(V[:,:NViews],axis=0)  # for empirical copula
        Cin = np.array(V[:,:NViews].argsort(axis=0),dtype=np.float64)+1 # for empirical copula
        Grid = np.arange(NSim)+1.0;
        
        C = zeros(Cin.shape)
        for k in range(NViews):
            f = interpolate.interp1d(Cin[:,k],Grid);
            C[:,k] = f(Grid)/(NSim+1);
            
        F = zeros((NSim,NViews)); 
        F_hat = zeros((NSim,NViews));
        F_tilda = zeros((NSim,NViews));
        V_tilda = zeros((NSim,NViews));
        
        for k in range(NViews): # determine the posterior margianl per view
            F[:,k]= Grid.T/(NSim+1);
            # Uniform view
            F_hat[:,k] = uniform.cdf(W[:,k], range_v[k,0], range_v[k,1])
            F_tilda[:,k] = (1-self.Conf[k])*F[:,k] + self.Conf[k]*F_hat[:,k]; #  weighted distribution
            # joint postorior
            f = interpolate.interp1d(F_tilda[:,k], W[:,k],fill_value='extrapolate');
            V_tilda[:,k] = f(C[:,k]);
            
            # Plot Simulation from views copula

            plt.figure('Simulation from views copula')
            plt.title('Simulation from views copula')
            plt.scatter(C[:,0],C[:,1])
            
            # Plot Cdf
            plt.figure(name)
            plt.title(name)
            plt.plot(W[:,0],F[:,0],'r', label='prior')
            plt.plot(W[:,0],F_hat[:,0], 'g', label= 'view')
            plt.plot(W[:,0],F_tilda[:,0], 'b', label= 'postorier')
            plt.legend()
        
    
    
