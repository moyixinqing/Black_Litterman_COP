# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:13:27 2019

@author: renxi
"""

import numpy as np
from numpy import r_, ones, zeros,eye, Inf
    
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from scipy.optimize import linprog


class PortfolioCVaR(object):
    """ The base class for PortfolioCVaR."""
    
    def __init__(self, data, NumPorts, quantile):
        self.data = data
        self.NumPorts = NumPorts
        self.quantile = quantile
  
    def objfCVaR(self, beta, q, N):
        """Computes vector c of the linear objective function f(x) = c'*x of the
         C-VaR Optimization problem 
         see: Uryasev, Rockafellar: Optimization of Conditional Value-at-Risk
         (1999)
         Input:
         beta -> confidence level
         q -> number of random vectors X1,...,Xq
         N -> nr of elements in vector Xi, i=1,...,q
         Output:
         cvec -> vector c of linear objective function f(x) = c'*x
        """
        cvec = r_[1, zeros((N)), 1/(1-beta)*1/q*ones((q))];
        return cvec

    def constCVaR(self, Sample,mu,Rstar):
        """Computes linear constrainst A*x <= b, Aeq*x = beq, lb <= x <= ub of the
         C-VaR Optimization problem see: Uryasev, Rockafellar: Optimization of 
         Conditional Value-at-Risk (1999)
         Input:
         Sample -> matrix of realizations of random variables X1,...,Xn
         mu -> vector of first moments of random variables X1,...,Xn
         Output:
         A -> matrix on the lefthanside of inequality constraints A*x <= b
         b -> vector on the righthanside of inequality constraints A*x <= b
         Aeq -> matrix on the lefthanside of equality constraints Aeq*x = beq
         beq -> vector on the righthanside of equality constraints Aeq*x = beq
         lb -> vector of lower bounds lb <= x
         ub -> vector of upper bounds x <= ub
        """
        [q,N] = Sample.shape;       
        A = zeros((q,1+N+q));
        A[:,0] = -1;
        A[:,1:N+1] = -Sample;
        A[:,N+1:] = -eye(q);       
        b = zeros((q,1));       
        Aeq = zeros((2,1+N+q));
        Aeq[0,1:N+1] = -mu.T;
        Aeq[1,1:N+1] = 1;       
        beq = zeros((2,1));
        beq[0] = -Rstar;
        beq[1] = 1;       
        lb = zeros(1+N+q);
        ub = r_[Inf,ones(N),Inf*ones(q)];           
        return [A,b,Aeq,beq,lb,ub]

    def CVaROpt_nonNormal(self):
    
        """This script demonstrates the use of function LINPROG on the basis of
         the C-VaR portfolio optimization problen given in:
         Uryasev, Rockafellar: Optimization of Conditional Value-at-Risk(1999)
        """
        # M: number of samples
        # N: number of assets
        [M,N] = self.data.shape;
        # sample mean
        mu = np.expand_dims(self.data.mean(axis=0), axis=0).T
        
        # confidence level
        beta = self.quantile;
        
        # target returns
        delta = (mu.max()-mu.min())/(self.NumPorts-1)
        Rstar = np.arange(mu.min(),mu.max()+delta,delta)
        
        # objective function vector
        f = self.objfCVaR(beta, M, N);
        
        # constraints matrices
        [A, b, Aeq, beq, lb, ub] = self.constCVaR(self.data, mu, Rstar[0]);
        
        CVaR    = zeros(self.NumPorts);
        weights = zeros((N,self.NumPorts));
        bnds = np.c_[lb,ub]
        
        bnds=((lb[0],ub[0]),)
        for i in range(1,len(lb)):
           bnds= bnds+((lb[i],ub[i]),)
        
        for i  in range(self.NumPorts):
            beq[0] = -Rstar[i];
            res = linprog(f, A, b, Aeq, beq, bounds= bnds);
            optimvar=res.x
            CVaR[i]=res.fun
            weights[:,i] = optimvar[1:N+1];
        
        return [weights, Rstar ,CVaR] 
    
    def plot(self,CVaR, Rstar,name):
        
        plt.figure(name)
        plt.title(name)
        plt.plot(CVaR, Rstar,'k', label = 'Mean-CVaR Frontier')
        plt.legend()
        plt.grid(True)
        plt.xlabel(r'$\beta$ -CVaR')
        plt.ylabel(r'$\mu$')
        plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
        plt.gca().set_xticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_xticks()]) 
        plt.show()
    
    def PlotCVaRFrontier(self, Portfolios,CVarvec,name):
        
        [xx,N] = Portfolios.shape;
        Data=np.cumsum(Portfolios,1);
        plt.figure(name)
            
        for n in range(N):
            #x = [1] + np.arange(1,xx+1).tolist() + [xx]
            x = [CVarvec[0]] + CVarvec.tolist() + [CVarvec[-1]]
            y = [0] + Data[:,N-n-1].tolist() +[0]
            #plt.fill(x, y, tuple(np.array([.9, .9, .9])- n%3 *np.array([.2, .2, .2])))
            plt.fill(x, y)
        plt.xlim((CVarvec[0], CVarvec[-1]))
        plt.ylim((0, np.max(Data)))
        plt.xlabel('portfolio # (risk propensity)')
        plt.ylabel('portfolio composition')
        plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
        plt.gca().set_xticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_xticks()]) 
    
        plt.title(name)
        plt.show()
      

