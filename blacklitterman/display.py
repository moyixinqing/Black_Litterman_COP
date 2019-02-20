import numpy as np
import pandas as pd
from blacklitterman.BlackLitterman import Blacklitterman


class display(Blacklitterman):
    """ """

    def __init__(self, title, assets, res):
        self.tau = res.tau
        self.P = res.P
        self.Q = res.Q
        self.Omega = res.Omega
        
        self.title = title
        self.assets = assets
        self.res = res
  
    def display(self):
        """Function to display the results of a black-litterman shrinkage

        Parameters
        ----------
        title :
            Displayed at top of output
        assets :
            List of self
        res :
            List of results structures from the bl function

        Returns
        -------

        
        """
        er = self.res.er
        w  = self.res.w
        lmbda = self.res.lmbda
      
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
        """ """
        line = ['Country']
        for p in range(len(self.P)):
            line = line + ['P{}'.format(p)]
        columns = line + ['mu','w*','w*-weq/(1+tau)']
        self.result = pd.DataFrame(columns= columns)
        self.result ['Country'] = np.array(self.assets)
        self.result = self.result.set_index('Country')
        for p in range(len(self.P)):
            self.result ['P{}'.format(p)] = np.array(self.P[p])*100
        self.result ['mu'] = np.array(self.res.er)*100
        self.result ['w*'] = np.array(self.res.w)*100
        self.result ['w*-weq/(1+tau)'] = np.array(self.res.change)*100
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
        lmbda = self.res.lmbda
        for l in lmbda:
            line = line + '{0:.3f}\t'.format(l[0])
            i = i + 1
        print(line)
        print('------------------------------------------------------------------\n')

    def disppd(self, hPi, pi):
        """

        Parameters
        ----------
        hPi :
            
        pi :
            

        Returns
        -------

        
        """
        print(self.title)
        weq = self.res.weq
        V  = self.res.V
        Er = self.res.er

        columns = ['Asset Name', 'Mkt Port (%)','Mean (%)','Std Dev (%)','BL Mean(%)','BL + View  Mean (%)', 'Investor Views'];
        result = pd.DataFrame(columns = columns)
        result['Asset Name'] = np.array(self.assets)
        result['Mkt Port (%)'] = 100*weq
        result['Mean (%)'] = 100*hPi
        result['Std Dev (%)'] = 100*np.sqrt(np.diag(V))
        result['BL Mean(%)'] = 100*pi
        result['BL + View  Mean (%)'] = 100*Er
        result['Investor Views'] = 100*self.P
        result = result.set_index('Asset Name')
        print(result.round(1).to_string())
        print('\n')
