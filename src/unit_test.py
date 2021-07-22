import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.dataset_torch import LFWImagePairList, LFWImageAlignPairList
from models.get_models_torch import CosfaceModel, SpherefaceModel, ArcfaceModel, ArcfaceModel2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np
import cv2

"""
ArcfaceModel2 acc is 0.9917
SpherefaceModel 0.9805  0.335
CosfaceModel 0.9913
"""


def run():

    dataset = LFWImagePairList(out_shape=(96,112))
    batch_size = 500
    datal = DataLoader(dataset, batch_size)

    cc_num = 0
    # threshold = 0.2245
    threshold = 0.335

    m = SpherefaceModel()
    

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
        print(pred.shape)
        print(sameflag.shape)
        cc_num += torch.sum(pred == sameflag )

    print( cc_num/len(dataset) )




if __name__ == '__main__':

    run()

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