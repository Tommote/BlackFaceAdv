
import os
import cv2
import numpy as np
import moxing as mox

from PIL import Image

from matlab_cp2tform import get_similarity_transform_for_cv2

class LFWImagePairList():
    def __init__(self, root_path='obs://mindspore-srq/dataset/archive/lfw-deepfunneled/lfw/' , transform=None, is_gray=False, alignment=True, out_shape=(96,112)):
        """
        This is the Dataset of LFW, the original image will be precoessed including 
        alignment , and will return a pair of image such as (), ()

        Args:
            root_path: the path of lfw data
            transform: transform torch version
            is_gray: process as gray image
            alignment: align the original image
            out_shape: the shape of output
        
        Return: a pair of image, and its type is Tensor [1, 3, out_shape[1], out_shape[0]]
                (img1, img1_),(img2, img2_)
        """

        self.root_path = root_path
        
        self.align, self.gray = alignment , is_gray

        self.out_shape = out_shape




        self.landmark = {}
        with mox.file.File('obs://mindspore-srq/test/lfw_landmark.txt', 'r') as f:
            landmark_lines = f.readlines()
        for line in landmark_lines:
            l = line.replace('\n','').split('\t')
            self.landmark[l[0]] = [int(k) for k in l[1:]]

        with mox.file.File('obs://mindspore-srq/test/pairs.txt','r') as f:
            self.pairs_lines = f.readlines()[1:]
        

    def __getitem__(self, index):
        
        p = self.pairs_lines[index].replace('\n', '').split('\t')

        if 3 == len(p):
            sameflag = 1
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        elif 4 == len(p):
            sameflag = 0
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        else:
            raise ValueError("WRONG LINE IN 'pairs.txt! ")

        # print(self.root_path+name1)

        img1 = self.read_img(self.root_path+name1 , cv2.IMREAD_UNCHANGED)
        img2 = self.read_img(self.root_path+name2 , cv2.IMREAD_UNCHANGED)

        if self.align:
            img1 = self.alignment( img1, self.landmark[name1] )
            img2 = self.alignment( img2, self.landmark[name2] )
        
        img1 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

        img1_ = img1.transpose(Image.FLIP_TOP_BOTTOM)
        img2_ = img2.transpose(Image.FLIP_TOP_BOTTOM)
        
#         img1 = np.array(img1).transpose(1,0,2)
#         img1_ = np.array(img1_).transpose(1,0,2)
#         img2 = np.array(img2).transpose(1,0,2)
#         img2_ = np.array(img2_).transpose(1,0,2)
        
        #return (img1.unsqueeze(0), img1_.unsqueeze(0)), (img2.unsqueeze(0), img2_.unsqueeze(0)), sameflag
        return np.array(img1), np.array(img1_), np.array(img2), np.array(img2_), sameflag
    
    def __len__(self):
        return len(self.pairs_lines)
    
    def alignment(self, src_img,src_pts):
        ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
            [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        # crop_size = (96, 112)
        crop_size = self.out_shape
        src_pts = np.array(src_pts).reshape(5,2)

        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)

        tfm = get_similarity_transform_for_cv2(s, r)
        face_img = cv2.warpAffine(src_img, tfm, crop_size)
        return face_img
    
    def read_img(self,path, mode):
        
        fc = mox.file.read( path, binary=True )
        fileNPArray=np.frombuffer(fc,np.uint8)

        x = cv2.imdecode( fileNPArray, mode )
        
        return x
        