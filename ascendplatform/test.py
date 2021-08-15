import mindspore.dataset as ds
from dataset import LFWImagePairList
import mindspore.dataset.vision.py_transforms as py_trans


dataset_generator = LFWImagePairList()
# print(dataset_generator[0][0].shape)
dataset = ds.GeneratorDataset(dataset_generator, column_names=['img1', 'img1_', 'img2', 'img2_', 'sameflag'], shuffle=False)

transforms_list = [py_trans.ToTensor(), py_trans.Normalize(mean=(0.5, ), std=(0.5,))]
dataset = dataset.map(operations=transforms_list, input_columns=['img1', 'img1_', 'img2', 'img2_'])

temp = dataset.create_dict_iterator()
data = next(temp)

print(data['img1'].shape)