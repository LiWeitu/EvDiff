"""Frozen public E2VID+ teacher, with raw outputs and recurrent sequence state."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

from training.teachers.e2vidplus import FlowNet
from training.utils import checkpoint_pickle


class E2VIDPlusTeacher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_encoders = model.num_encoders

    def reset_states(self):
        self.model.reset_states()

    def forward(self, events):
        height, width = events.shape[-2:]
        divisor = 2 ** self.num_encoders
        x = F.pad(events.float(), (0, (-width) % divisor, 0, (-height) % divisor))
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            image = self.model(x)['image']
        # EVREAL E2VID+: no event normalization or output normalization/clamping.
        return {'image': image[..., :height, :width].to(events.dtype)}


def load_teacher(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False,
                            pickle_module=checkpoint_pickle)
    arch = copy.deepcopy(checkpoint['config']['arch'])
    if arch['type'] != 'FlowNet':
        raise ValueError('Expected the public EVREAL E2VID+ FlowNet checkpoint')
    model = FlowNet(**arch['args'])
    # The public checkpoint also carries two flow channels; preserve its full head.
    if model.num_bins != 5 or model.unetflow.num_output_channels not in (1, 3):
        raise ValueError('Expected 5-bin E2VID+ with its reconstruction output')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    return E2VIDPlusTeacher(model).eval().requires_grad_(False)
