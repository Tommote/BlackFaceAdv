from models_mindspore.sphere_net import sphere20a 
from models_mindspore.cosface_net import sphere as cosface_net

from mindspore import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np



class SpherefaceModel_mind():

    def __init__(self, device='cpu') :
        """
        """
        
        model_path = 'pretrain_model/models_mind/sphere20a.ckpt'

        self.model = sphere20a(feature=True)

        load_param_into_net(self.model, load_checkpoint( model_path ))
        
        self.model.set_train(mode=False)
        # self.device = device
    
    def forward(self, img, img_):
        """
        """

        # img, img_ = img.to(self.device), img_.to(self.device)

        # ft = cat((self.model(img), self.model(img_)), 1)
        
        # return ft

        print( self.model( img ) )