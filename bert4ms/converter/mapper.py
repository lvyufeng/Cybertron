import collections
import json
import os

def build_weight_map(model_name, num_hidden_layers):
    """
    build weigh map from huggingface to mindspore
    """
    mapper_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), f'mappings/{model_name}.json')
    with open(mapper_path) as f:
        weight_map = json.load(f)

    layer_weight_map = collections.OrderedDict({k: v for k, v in weight_map.items() if '{}' in k})
    all_weight_map = collections.OrderedDict({k: v for k, v in weight_map.items() if '{}' not in k})
    for i in range(num_hidden_layers):
        for k, v in layer_weight_map.items():
            all_weight_map[k.format(i)] = v.format(i)
    
    return all_weight_map