import torch
from collections import OrderedDict
from pathlib import Path
import json

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt', encoding="utf-8") as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def show_param(model ,static_dict, print_param=False):
    for key in static_dict:
        frozen_layers = list(filter(lambda param : key in param[0], model.named_parameters()))
        for name, param in frozen_layers:
            print(name,param.requires_grad)
            if print_param:
                print(param, param.dtype)
                print(param.grad)
            else:
                print(param[0])

def load_lora_state_dict(state_dict, model, adapter_name="default"):
    for n, p in model.named_parameters():
        if adapter_name in n:
            name = "transformer." + n.replace(f".{adapter_name}", "")
            p.data.copy_(state_dict[name])
            state_dict.pop(name)
    if len(state_dict) > 0:
        print(f"Warning: {len(state_dict)} keys not loaded")
        print(state_dict.keys())

def get_trainable_param(model):
    train_param = []
    for n, p in model.named_parameters():
        if p.requires_grad:
            train_param.append(n)
    return train_param
