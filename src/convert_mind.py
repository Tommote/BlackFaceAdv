from mindspore import save_checkpoint
import torch

import mindspore

from models.get_models_torch import CosfaceModel, SpherefaceModel


from mindspore import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np


from mindspore.train.callback import ModelCheckpoint

def load_mindspore_torch( torch_model:torch.nn.Module, mind_model ):
    """
    This function is aimed to load mindspore model from the same torch 
    model, then save the mindespore model. This convert method is based 
    on the mindinsight awt convertor which can convert a torch suorce to
    corresponding the mindspore source. And this function can finish the next
    work that load the state dict form torch model.

    """
    state_dict_torch =  torch_model.state_dict()

    dict_m = mind_model.parameters_dict()
    # print(dict_m.keys())
    # print(state_dict_torch.keys())
    for name in dict_m.keys():

        param = dict_m[name]
        
        shape_mind = param.shape
        
        if name not in state_dict_torch:
            torch_name = map_name2(name)
            print("paramter {} not exist in torch model and {}".format(name, torch_name))
        else:
            print("the same name {}".format(name))
            torch_name = name
        # print(torch_name)
        shape_torch = state_dict_torch[torch_name].shape

        if shape_mind != shape_torch:
            raise 'The paramter {} error: mindspore shape is {} yet torch shape is {}'.format(name, shape_mind, shape_torch)
        # print(shape_mind)
        # print( state_dict_torch[torch_name].numpy().dtype )
        param.set_data(  mindspore.Tensor(  state_dict_torch[torch_name].numpy() , dtype=mindspore.float32) )


    return dict_m



def map_name(mind_name:str):

    temp_list = mind_name.split('.')
    n = len(temp_list)
    if temp_list[1]=='w':
        return temp_list[0]+'.weight'
    elif n==3 and temp_list[2]=='w':
        return temp_list[0]+'.'+temp_list[1]+'.weight'
    elif n==4 and temp_list[3]=='w':
        return temp_list[0]+'.'+temp_list[1]+'.'+temp_list[2]+'.weight'
    elif n==5 and temp_list[4]=='w':
        return temp_list[0]+'.'+temp_list[1]+'.'+temp_list[2]+'.'+temp_list[3]+'.weight'

def map_name2(mind_name:str):

    di = { 'moving_mean':'running_mean', 
            'w':'weight',
            'moving_variance':'running_var',
            'gamma':'weight',
            'beta':'bias' }

    temp_list = mind_name.split('.')

    res = ''

    for temp in temp_list:
        if temp in di:
           new_name = di[temp] 
        else:
            new_name = temp
        if res!='':
            res = res + '.'+new_name
        else:
            res = new_name
    
    return res

def map_name3(mind_name:str):

    di ={
        'batchnorm':{ 'moving_mean':'running_mean', 
            'w':'weight',
            'moving_variance':'running_var',
            'gamma':'bias',
            'beta':'weight' },
        
        'prelu':{

        },
        'conv':{

        }
    } 

    temp_list = mind_name.split('.')

    res = ''

    for temp in temp_list:
        if temp in di:
           new_name = di[temp] 
        else:
            new_name = temp
        if res!='':
            res = res + '.'+new_name
        else:
            res = new_name
    
    return res
def save_mindspore_model(net, params, path):

    load_param_into_net(net, params)
    # input = np.random.uniform(0.0, 1.0, size=shape).astype(np.float32)
    # export(net, Tensor(input), file_name=path, file_format='MINDIR')
    save_checkpoint(net, path)

def re_save():

    device = torch.device('cpu')
    dummy_input = torch.randn(1,3,96,112)

    model = SpherefaceModel(device)

    net = model.model

    torch.save( net, 'pretrain_model/sphFaceModel.pkl' )


def convert():

    device = torch.device('cpu')
    dummy_input = torch.randn(1,3,96,112)

    model = CosfaceModel(device)

    net = model.model

    print( net(dummy_input).shape )

    onnx_path = 'pretrain_model/models_onnx/spf_model.onnx'

    torch.onnx.export(net, dummy_input, onnx_path, input_names=['myinput'], output_names=['myoutput'])

    print('down')


if __name__=='__main__':

    convert()