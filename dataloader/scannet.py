# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
from collections import defaultdict
from tqdm import tqdm
import math
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.interpolate import RegularGridInterpolator
import torch
from torch.utils.data import Dataset
from dknet_ops import voxelization_idx


class ScanNet(Dataset):
    def __init__(self, split,           # Split enum(train, val)
                 root_dir,
                 scale=50,              # Defined by 1/resolution
                 max_crop_size=512,     # Max crop size in voxels for cropping
                 min_spatial_size=128,  # Min spatial size when clipping
                 max_num_points=250000, # Max number of points allows
                 ignore_index=-100,
                 **kwargs):
        self.root_dir, self.split = root_dir, split
        self.scale = scale
        self.max_crop_size = max_crop_size
        self.min_spatial_size = min_spatial_size
        self.max_num_points = max_num_points
        self.ignore_index = ignore_index

        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, return_aux=False):
        scene_id = self.data[idx]['scene_id']
        xyz = self.data[idx]['xyz']
        xyz -= xyz.mean(0)
        rgb = self.data[idx]['rgb'] / 127.5 - 1
        sem_label = self.data[idx]['sem_label']
        inst_label = self.data[idx]['inst_label']
        superpoint = self.data[idx]['superpoint']

        rot = np.eye(3)
        if self.split == 'train':
            xyz, rot = self.augment(xyz, return_rot=True)
            rgb += np.random.randn(3) * 0.1

        voxel = xyz * self.scale
        if self.split == 'train':
            voxel = self.elastic_distortion(voxel, 6 * self.scale // 50, 40 * self.scale / 50)
            voxel = self.elastic_distortion(voxel, 20 * self.scale // 50, 160 * self.scale / 50)
        voxel -= voxel.min(0)

        if self.split == 'train':
            voxel, valid = self.crop(voxel)
            xyz, rgb = xyz[valid], rgb[valid]
            sem_label, inst_label = sem_label[valid], inst_label[valid]
            superpoint = np.unique(superpoint[valid], return_inverse=True)[1]
        inst_label, shift_mapping = self.shift_inst_label(inst_label)
        inst_info, num_inst = self.get_instance_info(xyz, inst_label.astype(np.int32))

        scene_info = scene_id, voxel, xyz, rgb, sem_label, inst_label, superpoint, inst_info, num_inst
        if return_aux:
            return scene_info, (rot, shift_mapping)
        return *scene_info,

    def _load_data(self):
        scene_ids = [l[:-1] for l in open(os.path.join('dataloader', 'meta_data', f'scannetv2_{self.split}.txt')).readlines()]

        all_data = []
        for scene_id in tqdm(scene_ids, desc=f'Loading {self.split}-split'):
            minibatch = defaultdict(list)
            minibatch['scene_id'] = scene_id

            # Load ScanNet data
            filename_prefix = os.path.join(self.root_dir, 'processed_scans', scene_id)
            minibatch['xyz'] = np.load(filename_prefix+'_xyz.npy')
            minibatch['rgb'] = np.load(filename_prefix+'_rgb.npy')
            minibatch['sem_label'] = np.load(filename_prefix+'_sem_label.npy')
            minibatch['inst_label'] = np.load(filename_prefix+'_inst_label.npy')
            minibatch['superpoint'] = np.load(filename_prefix+'_superpoint.npy')
            all_data.append(minibatch)
        self.data = all_data

    def augment(self, xyz, jitter=True, rot=True, distort=True, return_rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        if return_rot:
            return np.matmul(xyz, m), m
        return np.matmul(xyz, m)

    def elastic_distortion(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x + g(x) * mag

    def crop(self, xyz):
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        max_crop_shape = np.array([self.max_crop_size] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_num_points):
            offset = np.clip(max_crop_shape - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < max_crop_shape).sum(1) == 3)
            max_crop_shape[:2] -= 32
        return xyz_offset[valid_idxs], valid_idxs

    def shift_inst_label(self, inst_label, mapping_size=999):
        unique_label = np.unique(inst_label)
        unique_label = unique_label[unique_label != self.ignore_index]
        shift_mapping = np.full(mapping_size, self.ignore_index, dtype=np.int32)
        shift_mapping[unique_label] = np.arange(unique_label.shape[0])

        valid = inst_label != self.ignore_index
        shifted_inst_label = np.full_like(inst_label, self.ignore_index)
        shifted_inst_label[valid] = shift_mapping[inst_label[valid]]
        return shifted_inst_label, shift_mapping

    def get_instance_info(self, xyz, inst_label):
        # Instance infos: (N, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_infos = np.ones((xyz.shape[0], 9), dtype=np.float32) * self.ignore_index
        num_inst = int(inst_label.max()) + 1
        for i in range(num_inst):
            indices_inst = np.where(inst_label == i)

            instance_info = instance_infos[indices_inst]
            xyz_inst = xyz[indices_inst]
            instance_info[:, 0:3] = xyz_inst.mean(0)
            instance_info[:, 3:6] = xyz_inst.min(0)
            instance_info[:, 6:9] = xyz_inst.max(0)
            instance_infos[indices_inst] = instance_info
        return instance_infos, num_inst

    def collate_fn(self, batch):
        inst_bias = 0
        superpoint_bias = 0
        batch_dict = defaultdict(list)
        for i, sample in enumerate(batch):
            scene_id, voxel, xyz, rgb, sem_label, inst_label, superpoint, inst_info, num_inst = sample
            batch_dict['scene_id'].append(scene_id)
            # Pad voxels with batch dimension
            batch_padding = torch.LongTensor(voxel.shape[0], 1).fill_(i)
            padded_voxel = torch.cat((batch_padding, torch.from_numpy(voxel).long()), dim=1)
            batch_dict['voxel'].append(padded_voxel)

            batch_dict['xyz'].append(torch.from_numpy(xyz))
            batch_dict['rgb'].append(torch.from_numpy(rgb))
            batch_dict['sem_label'].append(torch.from_numpy(sem_label.astype(int)))

            inst_label[inst_label != self.ignore_index] += inst_bias
            batch_dict['inst_label'].append(torch.from_numpy(inst_label.astype(int)))
            inst_bias += num_inst

            superpoint += superpoint_bias
            batch_dict['superpoint'].append(torch.from_numpy(superpoint))
            superpoint_bias += (superpoint.max() + 1)

            batch_dict['inst_info'].append(torch.from_numpy(inst_info))

        voxel = torch.cat(batch_dict['voxel'], 0)
        batch_dict['voxel'] = voxel                                                      # (N,4)
        batch_dict['xyz'] = torch.cat(batch_dict['xyz'], 0).float()                      # (N,3)
        batch_dict['rgb'] = torch.cat(batch_dict['rgb'], 0).float()                      # (N,3)

        batch_dict['sem_label'] = torch.cat(batch_dict['sem_label'], 0).long()           # (N,)
        inst_label = torch.cat(batch_dict['inst_label'], 0).long()
        batch_dict['inst_label'] = inst_label                                            # (N,)
        mask_label = torch.zeros((inst_bias, inst_label.shape[0]), dtype=torch.int)
        for i in range(mask_label.shape[0]):
            mask_label[i] = (inst_label == i).int()
        batch_dict['mask_label'] = mask_label                                            # (I,N)
        batch_dict['inst_info'] = torch.cat(batch_dict['inst_info'], 0).float()
        batch_dict['superpoint'] = torch.cat(batch_dict['superpoint'], 0).long()         # (N,)

        spatial_shape = np.clip((voxel.max(0)[0][1:] + 1).numpy(), self.min_spatial_size, None)
        batch_dict['spatial_shape'] = spatial_shape
        voxel_hash, p2v_map, v2p_map = voxelization_idx(voxel, len(batch), 4)            # 4 = mean
        batch_dict.update({'voxel_hash': voxel_hash, 'p2v_map': p2v_map, 'v2p_map': v2p_map})

        batch_dict['batch_size'] = len(batch)
        return batch_dict


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    root_dir = input()
    ds_train = ScanNet('train', root_dir)
    train_batch = next(iter(DataLoader(ds_train, batch_size=4, shuffle=True, collate_fn=ds_train.collate_fn)))
    ds_val = ScanNet('val', root_dir)
    val_batch = next(iter(DataLoader(ds_val, batch_size=4, shuffle=True, collate_fn=ds_val.collate_fn)))
