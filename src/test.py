import onnx
import torch
# import mindinsight

# from models.get_models_torch import CosfaceModel, SpherefaceModel, ArcfaceModel2

# device = torch.device('cpu')

# dummy_input = torch.randn(1, 3, 112, 112, device=device)

# model = ArcfaceModel2(device)

# net = model.model

# print(net(dummy_input).shape)

# torch.onnx.export(net, dummy_input, 'pretrain_model/models_onnx/ArcfaceModel2.onnx',verbose=False)

x = torch.tensor([[ [[1,2],[3,4]],[[5,6],[7,8]],[[9,12],[13,14]] ]])
print(x.shape)
print(x)
y = torch.flip(x, dims=[2])
print(y.shape)
print(y)