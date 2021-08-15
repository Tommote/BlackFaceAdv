from sphereNet import sphere20a 
from cosfaceNet import sphere as cosface
from arcfaceNet import iresnet18 
from utils import load_checkpoint_mox
import mindspore
from mindspore import export, load_param_into_net
from mindspore import Tensor
import numpy as np



class SpherefaceModel_mind():

    def __init__(self) :
        """
        """
        
        model_path = 'obs://mindspore-srq/pretrained_model/sphere20a.ckpt'

        self.model = sphere20a(feature=True)

        load_param_into_net(self.model, load_checkpoint_mox( model_path ))
        
        self.model.set_train(mode=False)
        self.ops = mindspore.ops.Concat(axis=1)
        # self.device = device
    
    def forward(self, img, img_):
        """
        """

        
        return ft

class CosfaceModel_mind():

    def __init__(self) :
        """
        """
        
        model_path = 'obs://mindspore-srq/pretrained_model/cosface_net.ckpt'

        self.model = cosface()

        load_param_into_net(self.model, load_checkpoint_mox( model_path ))
        
        self.model.set_train(mode=False)
        self.ops = mindspore.ops.Concat(axis=1)
 
    
    def forward(self, img, img_):
        """
        """

        ft = self.ops((self.model(img), self.model(img_)))
        
        return ft

class ArcfaceModel_mind():

    def __init__(self) :
        """
        """
        
        model_path = 'obs://mindspore-srq/pretrained_model/arcface_net_t.ckpt'

        self.model = iresnet18()

        load_param_into_net(self.model, load_checkpoint_mox( model_path ))
        
        self.model.set_train(mode=False)
        self.ops = mindspore.ops.Concat(axis=1)
 
    
    def forward(self, img, img_):
        """
        """

        ft = self.ops((self.model(img), self.model(img_)))
        
        return ft