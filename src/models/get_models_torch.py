import torch

from models.cosface_net import sphere as cosface_model
from models.sphere_net import sphere20a
from models.arcface_net import resnet_face18, resnet18
from models.arcface_net_2 import iresnet18

class CosfaceModel():

    def __init__(self, device='cuda:0') :
        """
        """
        
        model_path = 'pretrain_model/cosface_ACC99.28.pth'

        self.model = cosface_model()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model.to(device)
    
        self.device = device
    
    def forward(self, img, img_):
        """
        """
        with torch.no_grad():
            img, img_ = img.to(self.device), img_.to(self.device)
            
            ft = torch.cat((self.model(img), self.model(img_)), 1)
        return ft


class SpherefaceModel():

    def __init__(self, device='cuda:0') :
        """
        """
        
        model_path = 'pretrain_model/sphere20a_20171020.pth'

        self.model = sphere20a(feature=True)
        self.model.load_state_dict(torch.load(model_path), strict=False)
        self.model.eval()
        # self.model.feature = True
        self.model.to(device)
    
        self.device = device
    
    def forward(self, img, img_):
        """
        """
        with torch.no_grad():
            img, img_ = img.to(self.device), img_.to(self.device)

            ft = torch.cat((self.model(img), self.model(img_)), 1)
        return ft
    

class ArcfaceModel():

    def __init__(self, device='cuda:0') :
        """
        """
        
        model_path = 'pretrain_model/arcface_resnet18_110.pth'

        self.model = resnet_face18(use_se=False)
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() }
        self.model.load_state_dict(pretrained_dict)
        # self.load_model(self.model, model_path)
        self.model.eval()
        self.model.to(device)
    
        self.device = device
    
    def forward(self, img, img_):
        """
        """
        with torch.no_grad():
            img, img_ = img.to(self.device), img_.to(self.device)
            
            ft = torch.cat((self.model(img), self.model(img_)), 1)
        return ft

    def load_model(self, model, model_path):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


class ArcfaceModel2():

    def __init__(self, device='cuda:0') :
        """
        """
        
        model_path = 'pretrain_model/ms1mv3_arcface_r18_fp16.pth'

        self.model = iresnet18()
        pretrained_dict = torch.load(model_path, map_location=device)
        # pretrained_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items() }
        self.model.load_state_dict(pretrained_dict)
        self.model.eval()
        self.model.to(device)
    
        self.device = device
    
    def forward(self, img, img_):
        """
        """
        with torch.no_grad():
            img, img_ = img.to(self.device), img_.to(self.device)
            
            # x = torch.randn((500,3,112,112), dtype=torch.float32).to(self.device)
            # f1 = self.model(x)
            # print(f1.shape)

            ft = torch.cat((self.model(img), self.model(img_)), 1)
        return ft