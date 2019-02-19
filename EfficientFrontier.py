# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 19:43:59 2019

@author: renxi
"""

import numpy as np
from math import sqrt
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class EfficientFrontier(object):
    """The base class for EfficientFrontier."""
    
    def __init__(self,NumPortf, EstimatedCovariance, ExpectedValues):
        self.NumPortf = NumPortf
        self.EstimatedCovariance = EstimatedCovariance
        self.ExpectedValues = ExpectedValues
            
    def EfficientFrontier(self):
        """ This function returns the NumPortf x 1 vector expected returns, 
                               the NumPortf x 1 vector of volatilities and 
                               the NumPortf x NumAssets matrix of compositions 
         of NumPortf efficient portfolios whos returns are equally spaced along the whole range of the efficient frontier
        ExpectedValues = ExpValRets
        EstimatedCovariance = CovRets
        """
        NumAssets = self.EstimatedCovariance.shape[1];
        
        # determine return of minimum-risk portfolio
        FirstDegree=np.zeros((NumAssets,1));
        SecondDegree = self.EstimatedCovariance;
        Aeq=np.ones((1,NumAssets));
        beq=1;
        x0=1/NumAssets*np.ones((NumAssets,1));
        
        def fun(x):
           import numpy as np
           fun= 0.5*np.dot(np.dot(x.T,SecondDegree),x)+np.dot(FirstDegree.T,x)
           return fun
        def con(x):
            return np.dot(Aeq,x) - beq
        
        cons = ({'type': 'eq','fun': con})
        bnds = ((0, None),) * NumAssets
        MinVol_Weights = minimize(fun, x0, method='SLSQP',bounds=bnds,constraints=cons).x
        MinVol_Weights = np.expand_dims(MinVol_Weights,axis=0)
        #MinVol_Weights = quadprog(SecondDegree,FirstDegree,A,b,Aeq,beq,[],[],x0);
        MinVol_Return = np.dot(MinVol_Weights,self.ExpectedValues);
        
        # determine return of maximum-return portfolio
        MaxRet_Return = max(self.ExpectedValues);
        
        # slice efficient frontier in NumPortf equally thick horizontal sectors in the upper branch only
        TargetReturns=MinVol_Return + np.arange(self.NumPortf).T*(MaxRet_Return-MinVol_Return)/(self.NumPortf-1);
        
        # compute the NumPortf compositions and risk-return coordinates
        Composition = [];
        Volatility=[];
        ExpectedReturn=[];
        
        MinRet =min(self.ExpectedValues)[0];
        IndexMin = np.argmin(self.ExpectedValues)
        MaxRet =max(self.ExpectedValues);
        IndexMax = np.argmax(self.ExpectedValues)
        
        for i in range(self.NumPortf):
            # determine initial condition
            WMaxRet=(TargetReturns[0][i]-MinRet)/(MaxRet-MinRet);
            WMinRet=1-WMaxRet;
            x0=np.zeros((NumAssets,1));
            x0[IndexMax]=WMaxRet;
            x0[IndexMin]=WMinRet;
            
            # determine least risky portfolio for given expected return
            AEq = self.ExpectedValues;
            bEq = TargetReturns[0][i];
            
            #Weights = quadprog(SecondDegree,FirstDegree,A,b,AEq,bEq,[],[],x0).T;
            def fun(x):
                import numpy as np
                fun= 0.5*np.dot(np.dot(x.T,SecondDegree),x)+np.dot(FirstDegree.T,x)
                return fun
            def con(x):
                return np.dot(Aeq,x) - beq
        
            def con1(x):
                return np.dot(AEq.T,x) - bEq #np.expand_dims(bEq,axis=0).T 
        
            cons = ({'type': 'eq','fun': con},{'type': 'eq','fun': con1})
            bnds = ((0, 1),) * NumAssets
            Weights = minimize(fun, x0, method='SLSQP',bounds=bnds,constraints=cons).x
            Weights = np.expand_dims(Weights,axis=0)
        
            Composition.append((Weights[0]).tolist());
            Volatility=np.append(Volatility, sqrt(np.dot(np.dot(Weights,self.EstimatedCovariance),Weights.T)));
            ExpectedReturn=np.append(ExpectedReturn, TargetReturns[0][i]);
        Composition= np.array(Composition)  
        return [ExpectedReturn,Volatility, Composition] 
       
    def PlotFrontier(self,Portfolios,name):

        [xx,N]=Portfolios.shape;
        Data=np.cumsum(Portfolios,1);
        plt.figure(name)
            
        for n in range(N):
            x = [1] + np.arange(1,xx+1).tolist() + [xx]
            y = [0] + Data[:,N-n-1].tolist() +[0]
            #hold on
            #h=fill(x,y,[.9 .9 .9]-mod(n,3)*[.2 .2 .2]);
            plt.fill(x, y,tuple(np.array([.9, .9, .9])- n%3 *np.array([.2, .2, .2])))
        
        #set(gca,'xlim',[1 xx],'ylim',[0 max(max(Data))])
        plt.xlim((1, xx))
        plt.ylim((0, np.max(Data)))
        plt.xlabel('portfolio # (risk propensity)')
        plt.ylabel('portfolio composition')
        plt.title(name)
        plt.show()
