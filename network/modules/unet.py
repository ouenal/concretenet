# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

from collections import OrderedDict
import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch.modules import SparseModule


class ResidualBlock(SparseModule):
    def __init__(self, in_channels, out_channels, indice_key=None):
        super().__init__()
        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )
        self.conv_branch = spconv.SparseSequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        )

    def forward(self, x):
        identity = spconv.SparseConvTensor(x.features, x.indices, x.spatial_shape, x.batch_size)
        output = self.conv_branch(x)
        output = output.replace_feature(output.features + self.i_branch(identity).features)
        return output


class UBlock(nn.Module):
    def __init__(self, dim_list: list, repetitions, indice_key_id=1):
        super().__init__()
        self.dim_list = dim_list
        blocks = {'block{}'.format(i): ResidualBlock(dim_list[0], dim_list[0], indice_key='subm{}'.format(indice_key_id)) for i in range(repetitions)}
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(dim_list) > 1:
            self.conv = spconv.SparseSequential(
                nn.BatchNorm1d(dim_list[0]),
                nn.ReLU(),
                spconv.SparseConv3d(dim_list[0], dim_list[1], kernel_size=2, stride=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )
            self.unet = UBlock(dim_list[1:], repetitions, indice_key_id=indice_key_id+1)
            self.deconv = spconv.SparseSequential(
                nn.BatchNorm1d(dim_list[1]),
                nn.ReLU(),
                spconv.SparseInverseConv3d(dim_list[1], dim_list[0], kernel_size=2, bias=False, indice_key='spconv{}'.format(indice_key_id))
            )

            blocks_tail = {}
            for i in range(repetitions):
                blocks_tail['block{}'.format(i)] = ResidualBlock(dim_list[0] * (2 - i), dim_list[0], indice_key='subm{}'.format(indice_key_id))
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, x):
        output = self.blocks(x)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)
        if len(self.dim_list) > 1:
            output_decoder = self.conv(output)
            output_decoder = self.unet(output_decoder)
            output_decoder = self.deconv(output_decoder)
            output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))
            output = self.blocks_tail(output)
        return output
