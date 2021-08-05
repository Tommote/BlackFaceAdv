import os

import mindspore
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.dataset_torch import LFWImagePairList, LFWImageAlignPairList
from models.get_models_torch import CosfaceModel, SpherefaceModel, ArcfaceModel, ArcfaceModel2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2


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

""" 
ArcfaceModel2 acc is 0.9917
SpherefaceModel 0.9805  0.335
CosfaceModel 0.9913
"""

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
    save_mindspore_model( net=model_mind,params=params, path= 'pretrain_model/models_mind/arcface_net.ckpt')

    # sp_model = SpherefaceModel_mind()
    print( params['features.beta'].asnumpy() )

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

        print( cc_num.item() /len(dataset) )
    print( cc_num.item()/len(dataset) )

def cal_distance(f1, f2):

    t1 = np.sum(f1*f2, axis=1)

    t2 = np.sqrt(np.sum(f1*f1,axis=1))
    t3 = np.sqrt(np.sum(f2*f2, axis=1))

    return t1/( t2*t3 + 1e-5)

if __name__ == '__main__':

    run_sp()

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