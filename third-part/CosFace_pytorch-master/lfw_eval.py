from PIL import Image
import numpy as np
import cv2
from matlab_cp2tform import get_similarity_transform_for_cv2
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import net

landmark = {}
with open('/home/srq/pythonworkplace/Black-box-AdvAttack-face/third-part/sphereface_pytorch-master/data/lfw_landmark.txt') as f:
    landmark_lines = f.readlines()
for line in landmark_lines:
    l = line.replace('\n','').split('\t')
    landmark[l[0]] = [int(k) for k in l[1:]]


def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

def extractDeepFeature(img, points ,model, is_gray):
    
    img = alignment(img, points)
    # print(img.shape)
    # img = img.transpose(2, 0, 1).reshape((3,112,96))
    # img = (img-127.5)/128.0
    # print(img.shape)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if is_gray:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5,), std=(0.5,))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
    img, img_ = transform(img), transform(F.hflip(img))

    img, img_ = img.unsqueeze(0).to('cuda'), img_.unsqueeze(0).to('cuda')

    ft = torch.cat((model(img), model(img_)), 1)[0].to('cpu')
    return ft


def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def eval(model, model_path=None, is_gray=False):
    predicts = []
    model.load_state_dict(torch.load(model_path))
    model.eval()
    folds = KFold(n=6000, n_folds=10)
    root = '/home/srq/datasets/LFW/lfw_all/'
    with open('/home/srq/datasets/LFW/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]

    with torch.no_grad():
        for i in range(6000):
            p = pairs_lines[i].replace('\n', '').split('\t')

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

            # with open(root + name1, 'rb') as f:
            #     img1 =  Image.open(f).convert('RGB')
            # with open(root + name2, 'rb') as f:
            #     img2 =  Image.open(f).convert('RGB')
            img1 = cv2.imread(root+name1 , cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(root+name2 , cv2.IMREAD_UNCHANGED)

            f1 = extractDeepFeature(img1, landmark[name1], model, is_gray)
            f2 = extractDeepFeature(img2, landmark[name2], model, is_gray)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, distance, sameflag))

    accuracy = []
    thd = []
    
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts = np.array( list( map(lambda line: line.strip('\n').split(), predicts)))
    
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))

    return np.mean(accuracy), predicts


if __name__ == '__main__':
    _, result = eval(net.sphere().to('cuda'), model_path='/home/srq/pythonworkplace/Black-box-AdvAttack-face/pretrain_model/ACC99.28.pth')
    np.savetxt("result.txt", result, '%s')
    # predicts = []
    # predicts.append('{}\t{}\t{}\t{}\n'.format('Gordon_Brown/Gordon_Brown_0010.jpg', 'Gordon_Brown/Gordon_Brown_0013.jpg', 0.6565602421760559, 1))

    # predicts = list(map(lambda line: line.strip('\n').split(), predicts))
    # predicts = np.array(predicts)
    # print(predicts.shape)