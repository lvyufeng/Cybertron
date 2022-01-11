import torch
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint
from mapper import build_weight_map

def convert_state_dict(pth_file):
    ms_ckpt = []
    state_dict = torch.load(pth_file)
    # weight_map = build_weight_map('bert', 12)
    for k, v in state_dict.items():
        if 'embeddings' in k:
            k = k.replace('weight', 'embedding_table')
        if 'LayerNorm' in k:
            k = k.replace('LayerNorm', 'layer_norm')
        if 'self' in k:
            k = k.replace('self', 'self_attn')
        print(k)
        ms_ckpt.append({'name': k, 'data':Tensor(v.numpy())})
    save_checkpoint(ms_ckpt, pth_file + '.ckpt')    
    pass

convert_state_dict('/home/lvyufeng/Downloads/bert-base-uncased/pytorch_model.bin')