import os

import mindspore
import mindspore.nn as mind_nn
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from utils.dataset_torch import LFWImagePairList, LFWImageAlignPairList
from models.get_models_torch import CosfaceModel, SpherefaceModel, ArcfaceModel, ArcfaceModel2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2
import math
from torchvision.utils import make_grid
from models_mindspore.sphere_net import sphere20a
from models_mindspore.arcface_net import iresnet18
from models_mindspore.cosface_net import sphere as cosface_net
from models_mindspore.get_models_mindspore import SpherefaceModel_mind
from convert_mind import load_mindspore_torch, save_mindspore_model
# from utils.dataset_mind import LFWImagePairList
from utils.dataset_torch import LFWImagePairList
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as c_vision
from mindspore.dataset.vision import Inter

import mindspore.dataset.vision.py_transforms as py_trans

from attack.evolution_black_attack import EvolutionaryAttack

""" 
ArcfaceModel2 acc is 0.9917
SpherefaceModel 0.9805  0.335
CosfaceModel 0.9913
"""

def run_attack():

    device = torch.device('cpu')

    dataset = LFWImagePairList(out_shape=(112,112))
    batch_size = 500
    datal = DataLoader(dataset, batch_size)

    cc_num = 0
    threshold = 0.2245
    # threshold = 0.335

    m = ArcfaceModel2(device)

    ea = EvolutionaryAttack(threaten_model=m)

    xs = next(iter(datal))[0][0][120].unsqueeze(0)

    # print(xs.shape)
    

    save_img(xs[0], 'ori.png')
    x_adv = ea.gen_advExample(x_ori=xs, T=1000)

    print(x_adv.shape)
    save_img(x_adv[0], 'adv.png')

    f1 = m.forward(xs, torch.flip(xs, dims=[2]))
    f2 = m.forward(x_adv, torch.flip(x_adv, dims=[2]))

    distance = torch.sum(f1*f2, dim=1) / (f1.norm(dim=1) * f2.norm(dim=1) + 1e-5)
    print(distance)

def save_img(x, name):

    def unnormalize(tensor, mean, std, inplace: bool = False) :
        """Unnormalize a tensor image with mean and standard deviation.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation inplace.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('Input tensor should be a torch tensor. Got {}.'.format(type(tensor)))

        if tensor.ndim < 3:
            raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.size() = '
                            '{}.'.format(tensor.size()))

        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        tensor.mul_(std).add_(mean)
        return tensor
    


    def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
        '''
        Converts a torch Tensor into an image Numpy array
        Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
        Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
        '''
        tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
        tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
        n_dim = tensor.dim()
        if n_dim == 4:
            n_img = len(tensor)
            img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        elif n_dim == 3:
            img_np = tensor.detach().numpy()
            #img_np = tensor.detach().numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        elif n_dim == 2:
            img_np = tensor.detach().numpy()
            #img_np = tensor.detach().numpy()
        else:
            raise TypeError(
                'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
            # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
        return img_np.astype(out_type)

    x = unnormalize(x,mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    cv2.imwrite('output/pics/'+name, tensor2img(x))

def run_opt():

    inplanes = 128

    m1 = torch.nn.BatchNorm2d(inplanes, eps=1e-05)

    m2 = mind_nn.BatchNorm2d( inplanes,  eps=1e-05, momentum=0.9)


    x = np.random.randn(12,128,112,112)

    x1 = torch.from_numpy(x)
    x2 = mindspore.Tensor(x)

    p1 = m1.state_dict()
    p2 = m2.parameters_dict()

    print(p2.keys())

    print(torch.flatten(x1,1).shape)
    print(mindspore.ops.operations.Flatten()(x2).shape)
    print(x2.shape)

    # print(p1['weight'].shape)
    # print(p1['bias'].shape)
    # print(p1['running_mean'].shape)
    # print(p1['running_var'].shape)

    # print(p2['mean'])
    # print(p2['variance'])
    # print(p2['gamma'])
    # print(p2['beta'])

    # print(m1(x1).shape)

    print( "dfsdf".find('fsdf') )

def run_dataset():

    dataset_generator = LFWImagePairList()
    # print(dataset_generator[0][0][0])


    dataset = ds.GeneratorDataset(dataset_generator, ["img1","img1_","img2","img2_", "label"], shuffle=False)

    transforms_list = [py_trans.ToTensor(), py_trans.Normalize(mean=(0.5, ), std=(0.5,))]
    dataset = dataset.map(operations=transforms_list)

    temp = dataset.create_dict_iterator()
    data = next(temp)

    print(data['img2_'].shape)
    print(data['label'])

def run_sp():

    model_mind = iresnet18()
    model_torch = ArcfaceModel2(device='cpu').model
    # print(model_torch.state_dict().keys())
    params = load_mindspore_torch(torch_model=model_torch, mind_model=model_mind)
    # print(params.keys())
    save_mindspore_model( net=model_mind,params=params, path= 'pretrain_model/models_mind/arcface_net_t.ckpt')

    # sp_model = SpherefaceModel_mind()
    # print( params['features.beta'].asnumpy() )

    # sp_model.forward( input , input)

def run1():

    model_mind = sphere20a(feature=True)
    dict_m = model_mind.parameters_dict()

    temp = mindspore.Tensor( np.random.randn(64,3,3,3) , dtype=mindspore.float32)

    print(dict_m['conv1_1.weight'].set_data(temp))

    model_torch = SpherefaceModel(device='cpu').model

    # print( model_torch.state_dict().keys())
    # print(dict_m)

def run():

    device = torch.device('cpu')

    dataset = LFWImagePairList(out_shape=(112,112))
    batch_size = 500
    datal = DataLoader(dataset, batch_size)

    cc_num = 0
    threshold = 0.2245
    # threshold = 0.335

    m = ArcfaceModel2(device)
    

    for (img1, img1_), (img2, img2_), sameflag in datal:
        # (img1, img1_), (img2, img2_), sameflag = dataset[i*batch_size:(i+1)*batch_size]

        img1_ = torch.flip(img1, dims=[2])
        img2_ = torch.flip(img2, dims=[2])

        f1 = m.forward(img1,img1_)
        f2 = m.forward(img2, img2_)
        print(f1.shape)
        distance = torch.sum(f1*f2, dim=1) / (f1.norm(dim=1) * f2.norm(dim=1) + 1e-5)
        
        pred = torch.zeros( (distance.shape[0]) )


        pred[ distance>threshold ] = 1

        # if distance< threshold:
        #     pred = 0
        # else:
        #     pred = 1
        cc_num += torch.sum(pred == sameflag )

        print( cc_num.item() / pred.shape[0] )
    print( cc_num.item()/len(dataset) )

def cal_distance(f1, f2):

    t1 = np.sum(f1*f2, axis=1)

    t2 = np.sqrt(np.sum(f1*f1,axis=1))
    t3 = np.sqrt(np.sum(f2*f2, axis=1))

    return t1/( t2*t3 + 1e-5)

if __name__ == '__main__':

    run_attack()

    # image = np.random.randint(0,255,(128,128,1))

    # image = np.dstack((image, np.fliplr(image)))
    # image = image.transpose((2, 0, 1))
    # image = image[:, np.newaxis, :, :]
    # # image = np.concatenate((image, image), axis=0)

    # print(image.shape)
    # print(cv2.IMREAD_GRAYSCALE)

    # root = '/home/srq/datasets/LFW/lfw-align-128/'
    # with open('src/utils/lfw_test_pair.txt') as f:
    #     pairs_lines = f.readlines()
    # acc = 0
    # new_pair = []
    # for x in pairs_lines:
    #     y = x.split()
    #     name1 = y[0]
    #     name2 = y[1]

    #     img1 = cv2.imread(root+name1)
    #     img2 = cv2.imread(root+name2)

    #     if img1 is not None and img2 is not None:
    #         acc+=1
    #         new_pair.append( (name1, name2, y[2]) )
    
    # with open('src/utils/lfw_test_pair_new.txt', 'w+') as f:

    #     for x in new_pair:
    #         f.write( x[0] + ' '+x[1]+' '+ x[2]+'\n') 

    # print(len(pairs_lines))
    # print(acc)