import torch
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint
from mapper import build_weight_map

def convert_state_dict(pth_file):
    ms_ckpt = []
    state_dict = torch.load(pth_file)
    weight_map = build_weight_map('bert', 12)
    for k, v in state_dict.items():
        if k not in weight_map.keys():
            continue
        ms_ckpt.append({'name': weight_map[k], 'data':Tensor(v.numpy())})
    save_checkpoint(ms_ckpt, pth_file + '.ckpt')    
    pass

convert_state_dict('/root/bert-base-uncased-pytorch_model.bin')