# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFilterLayer(nn.Module):
    def __init__(self, filter_size, stride=1, pad=0, flip_filters=False, grouping=False):
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.pad = pad
        self.flip_filters = flip_filters
        self.grouping = grouping

    def forward(self, _input, **kwargs):
        points = _input[0]
        filters = _input[1]

        input_channel = points.shape[-1]
        if points.dim() == 2:
            points = points.unsqueeze(0).permute(0, 2, 1)
            for i, filter_num in enumerate(self.filter_size):
                filter_size = input_channel * filter_num
                filter_weight = filters[:filter_size].view(filter_num, input_channel, 1)
                filters = filters[filter_size:]
                filter_bias = filters[:filter_num].view(-1)
                filters = filters[filter_num:]

                points = F.conv1d(points, filter_weight, filter_bias, padding = self.pad)
                if i < (len(self.filter_size) -1):
                    points = F.relu(points)

                input_channel = filter_num
        else:
            n_mask = points.shape[1]
            num_instances = points.shape[0]
            points = points.permute(0, 2, 1).reshape(1, -1, n_mask)
            for i, filter_num in enumerate(self.filter_size):
                filter_size = input_channel * filter_num
                filter_weight = filters[:, :filter_size].reshape(num_instances * filter_num, input_channel, 1)
                filters = filters[:, filter_size:]
                filter_bias = filters[:, :filter_num].reshape(-1)
                filters = filters[:, filter_num:]

                points = F.conv1d(points, filter_weight, filter_bias, padding = self.pad, groups=num_instances)
                if i < (len(self.filter_size) -1):
                    points = F.relu(points)

                input_channel = filter_num
            points = points.squeeze()

        output = torch.sigmoid(points)
        return output
