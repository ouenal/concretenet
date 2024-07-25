# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch.nn as nn

from network.modules.dknet import DKNet
from network.modules.mpnet import MPNet
from network.modules.baf import BAFModule


class ConcreteNet(nn.Module):
    def __init__(self, visual_config, verbal_config, fusion_config):
        super().__init__()
        self.visual_backbone = DKNet(**visual_config)
        self.verbal_backbone = MPNet(**verbal_config)
        self.fusion_module = BAFModule(input_dim=visual_config['output_dim'],
                                       dim=verbal_config['output_dim'],
                                       **fusion_config)

    def forward(self, input_dict, stage=-1):
        '''
        Training stages:
        Stage  0: pretraining the point cloud encoder
        Stage  1: pretraining the candidate generation
        Stage  2: pretraining the instance backbone
        Stage  3: end to end trainining
        Stage -1: inference, avoid augmentation
        '''
        visual_ret = self.visual_backbone(input_dict, stage)
        if (stage > 0 and stage < 3) or 'description' not in input_dict:
            return visual_ret
        verbal_ret = self.verbal_backbone(input_dict, stage > 0)
        fusion_ret = self.fusion_module(input_dict, visual_ret, verbal_ret)
        return {**visual_ret, **verbal_ret, **fusion_ret}

    def decode_output(self, output_dict):
        ret = self.visual_backbone.decode_output(output_dict)
        if 'lang_output' in output_dict:
            ret.update(self.fusion_module.decode_output(output_dict))
        return ret
