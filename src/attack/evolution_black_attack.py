import numpy as np
import torch

import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision.transforms import functional 

class EvolutionaryAttack:
    """
    The implementation of evolutionary attack
    """

    def __init__(self, threaten_model,  attack_mode='dodging',threshold=0.2245,cfg=None):
        """
        """
        
        self.threaten_model = threaten_model
        self.attack_mode = attack_mode

        self.threshold = threshold

        self.hyper_param = {

          'm':  3*45*45,
          'k':  (3*45*45)//20,
          'cc': 0.01,
          'ccov': 0.001,
          'sigma': 0.01

        } 

    
    def gen_advExample(self, x_ori, x_target=None, T=1000):

        # init

        x_shape = x_ori.shape

        m = self.hyper_param['m']
        k = self.hyper_param['k']
        C = torch.eye(self.hyper_param['m'])
        pc = 0
        sigma = self.hyper_param['sigma']
        mu = 1
        cc = self.hyper_param['cc']
        ccov = self.hyper_param['ccov']

        success_rate = 0

        if self.attack_mode == 'dodging':
            x_adv = torch.randn_like(x_ori)
        else:
            raise ValueError('attack mode {} is still not be implemented ')
        

        for t in range(T):

            sampler = MultivariateNormal(loc=torch.zeros(m), 
                                covariance_matrix=(sigma**2)*C)
            
            z = sampler.sample()

            zeroIdx = np.argsort(-C.diagonal())[k:]
            z[zeroIdx] = 0
            
            z = z.reshape([1,3,45,45]) 
            z_ = F.interpolate(z,(x_shape[2],x_shape[3]),mode = 'bilinear')
            z_ = z_ + mu * (x_ori - x_adv)

            loss1 = self._loss(x_ori, x_adv+z_)
            loss2 = self._loss(x_ori, x_adv)

            if loss1<loss2:
                x_adv = x_adv + z_
                pc = (1 - cc) * pc + np.sqrt(2*(2-cc)) * z.reshape(-1)/sigma
                C[range(m),range(m)] = (1 - ccov)*C.diagonal() + ccov *(pc)**2
                success_rate += 1
            
            if t % 10 == 0 :
        
                mu = mu* np.exp(success_rate/10 - 1/5)
                success_rate = 0
        
        return x_adv

            
    def _loss(self, input1, input2):

        pred = self._pred(input1, input2)

        D = torch.norm(input1-input2)
        C = torch.zeros((input1.shape[0]))
        C[ pred==1 ].fill_(float('inf'))

        return D+C

    def _pred(self, input1, input2):
        
        with torch.no_grad():
            f1 = self.threaten_model.forward(input1, torch.flip(input1, dims=[2]))
            f2 = self.threaten_model.forward(input2, torch.flip(input2, dims=[2]))
            
            distance = torch.sum(f1*f2, dim=1) / (f1.norm(dim=1) * f2.norm(dim=1) + 1e-5)

        pred = torch.zeros( (distance.shape[0]) )
        pred[ distance>self.threshold ] = 1

        return pred

