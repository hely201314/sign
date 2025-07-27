from slr_network import SLRModel
import numpy as np
import torch
from collections import OrderedDict
import utils
from thop import profile

device_id = 0
dataset = 'phoenix2014'
dict_path = f'./preprocess/{dataset}/gloss_dict.npy'
gloss_dict = np.load(dict_path, allow_pickle=True).item()
model_weights = '/remote-home/cs_cs_heli/AdaptSign/work_dir/best/dev_18.70_epoch49_model.pt'

def reshape_transform(tensor, height=14, width=14):
    # 去掉类别标记
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result





device = utils.GpuDataParallel()
device.set_device(device_id)
model = SLRModel( num_classes=1296, c2d_type='ViT-B/16', conv_type=2, use_bn=1, gloss_dict=gloss_dict,
            loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0},   )
state_dict = torch.load(model_weights)['model_state_dict']
state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
model.load_state_dict(state_dict, strict=False)
model = model.to(device.output_device)
model.cuda()

model.eval()


# for name in model.named_modules():
#     print(name)

flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))