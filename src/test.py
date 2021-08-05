import onnx
import torch
import mindinsight

from models.get_models_torch import CosfaceModel, SpherefaceModel, ArcfaceModel2

device = torch.device('cpu')

dummy_input = torch.randn(1, 3, 112, 112, device=device)

model = ArcfaceModel2(device)

net = model.model

print(net(dummy_input).shape)

torch.onnx.export(net, dummy_input, 'pretrain_model/models_onnx/ArcfaceModel2.onnx',verbose=False)