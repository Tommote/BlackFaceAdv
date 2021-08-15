from mindspore._checkparam import check_input_data, Validator
from mindspore.train.checkpoint_pb2 import Checkpoint
import numpy as np
from mindspore.common.parameter import Parameter
import moxing 
from mindspore.common import dtype as mstype
from mindspore.common.tensor import Tensor
tensor_to_np_type = {"Int8": np.int8, "Uint8": np.uint8, "Int16": np.int16, "Uint16": np.uint16,
                     "Int32": np.int32, "Uint32": np.uint32, "Int64": np.int64, "Uint64": np.uint64,
                     "Float16": np.float16, "Float32": np.float32, "Float64": np.float64, "Bool": np.bool_}

tensor_to_ms_type = {"Int8": mstype.int8, "Uint8": mstype.uint8, "Int16": mstype.int16, "Uint16": mstype.uint16,
                     "Int32": mstype.int32, "Uint32": mstype.uint32, "Int64": mstype.int64, "Uint64": mstype.uint64,
                     "Float16": mstype.float16, "Float32": mstype.float32, "Float64": mstype.float64,
                     "Bool": mstype.bool_}

def load_checkpoint_mox(ckpt_file_name, net=None, strict_load=False, filter_prefix=None, dec_key=None, dec_mode="AES-GCM"):
    """
    Loads checkpoint info from a specified file.

    Args:
        ckpt_file_name (str): Checkpoint file name.
        net (Cell): Cell network. Default: None
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                           in the param_dict into net with the same suffix and load
                           parameter with different accuracy. Default: False.
        filter_prefix (Union[str, list[str], tuple[str]]): Parameters starting with the filter_prefix
            will not be loaded. Default: None.
        dec_key (Union[None, bytes]): Byte type key used for decryption. If the value is None, the decryption
                                      is not required. Default: None.
        dec_mode (str): This parameter is valid only when dec_key is not set to None. Specifies the decryption
                        mode, currently supports 'AES-GCM' and 'AES-CBC'. Default: 'AES-GCM'.

    Returns:
        Dict, key is parameter name, value is a Parameter.

    Raises:
        ValueError: Checkpoint file is incorrect.

    Examples:
        >>> from mindspore import load_checkpoint
        >>>
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = load_checkpoint(ckpt_file_name, filter_prefix="conv1")
        >>> print(param_dict["conv2.weight"])
        Parameter (name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True
    """
    #ckpt_file_name, filter_prefix = _check_checkpoint_param(ckpt_file_name, filter_prefix)
    dec_key = Validator.check_isinstance('dec_key', dec_key, (type(None), bytes))
    dec_mode = Validator.check_isinstance('dec_mode', dec_mode, str)
    print("Execute the process of loading checkpoint files.")
    checkpoint_list = Checkpoint()

    try:

        with moxing.file.File(ckpt_file_name, "rb") as f:
            pb_content = f.read()
        
        checkpoint_list.ParseFromString(pb_content)
    except BaseException as e:

        print("Failed to read the checkpoint file `%s`. The file may be encrypted, please pass in the "
                         "correct dec_key.", ckpt_file_name)

        raise ValueError(e.__str__())

    parameter_dict = {}
    try:
        param_data_list = []
        for element_id, element in enumerate(checkpoint_list.value):
            if filter_prefix is not None and _check_param_prefix(filter_prefix, element.tag):
                continue
            data = element.tensor.tensor_content
            data_type = element.tensor.tensor_type
            np_type = tensor_to_np_type[data_type]
            ms_type = tensor_to_ms_type[data_type]
            element_data = np.frombuffer(data, np_type)
            param_data_list.append(element_data)
            if (element_id == len(checkpoint_list.value) - 1) or \
                    (element.tag != checkpoint_list.value[element_id + 1].tag):
                param_data = np.concatenate((param_data_list), axis=0)
                param_data_list.clear()
                dims = element.tensor.dims
                if dims == [0]:
                    if 'Float' in data_type:
                        param_data = float(param_data[0])
                    elif 'Int' in data_type:
                        param_data = int(param_data[0])
                    parameter_dict[element.tag] = Parameter(Tensor(param_data, ms_type), name=element.tag)
                elif dims == [1]:
                    parameter_dict[element.tag] = Parameter(Tensor(param_data, ms_type), name=element.tag)
                else:
                    param_dim = []
                    for dim in dims:
                        param_dim.append(dim)
                    param_value = param_data.reshape(param_dim)
                    parameter_dict[element.tag] = Parameter(Tensor(param_value, ms_type), name=element.tag)

        print("Loading checkpoint files process is finished.")

    except BaseException as e:
        print("Failed to load the checkpoint file `%s`.", ckpt_file_name)
        raise RuntimeError(e.__str__())

    if not parameter_dict:
        raise ValueError(f"The loaded parameter dict is empty after filtering, please check filter_prefix.")

    if net is not None:
        load_param_into_net(net, parameter_dict, strict_load)

    return parameter_dict


def cal_distance(f1, f2):

    t1 = np.sum(f1*f2, axis=1)

    t2 = np.sqrt(np.sum(f1*f1,axis=1))
    t3 = np.sqrt(np.sum(f2*f2, axis=1))

    return t1/( t2*t3 + 1e-5)