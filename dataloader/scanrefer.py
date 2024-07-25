# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import json
from collections import defaultdict
from tqdm import tqdm
import math
import numpy as np
import torch
from dknet_ops import voxelization_idx

try:
    from dataloader.scannet import ScanNet
except:
    from scannet import ScanNet


class ScanRefer(ScanNet):
    def __init__(self, split,           # Split enum(train, val)
                 root_dir,
                 scale=50,              # Defined by 1/resolution
                 max_crop_size=512,     # Max crop size in voxels for cropping
                 min_spatial_size=128,  # Min spatial size when clipping
                 max_num_points=250000, # Max number of points allows
                 minibatch_size=32,     # Number of referrals per minibatch
                 ignore_index=-100):
        self.root_dir, self.split = root_dir, split
        self.scale = scale
        self.max_crop_size = max_crop_size
        self.min_spatial_size = min_spatial_size
        self.max_num_points = max_num_points
        self.ignore_index = ignore_index
        self.minibatch_size = minibatch_size

        self._load_scanrefer()
        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene_info, aux = super().__getitem__(idx, return_aux=True)

        object_id = np.array(self.data[idx]['object_id'])
        description = self.data[idx]['description']
        object_name = self.data[idx]['object_name']
        cam_label = np.stack([c['position'] for c in self.data[idx]['cam_label']], axis=0)

        rot, shift_mapping = aux
        object_id = shift_mapping[object_id]
        cam_label = np.matmul(cam_label, rot)
        return *scene_info, object_id, description, object_name, cam_label

    def _load_scanrefer(self, load_camera=True):
        scanrefer_path = os.path.join(self.root_dir, 'scanrefer', f'ScanRefer_filtered_{self.split}.json')
        scanrefer = json.load(open(scanrefer_path))

        # Reorganize scanrefer into minibatches
        scanrefer_new = []
        scanrefer_new_scene = []
        unique_scene_ids = []
        scene_id = ''
        for data in scanrefer:
            if scene_id != data['scene_id']:
                scene_id = data['scene_id']
                unique_scene_ids.append(scene_id)
                if len(scanrefer_new_scene) > 0:
                    scanrefer_new.append(scanrefer_new_scene)
                scanrefer_new_scene = []
            if len(scanrefer_new_scene) >= self.minibatch_size:
                scanrefer_new.append(scanrefer_new_scene)
                scanrefer_new_scene = []
            scanrefer_new_scene.append(data)
        scanrefer_new.append(scanrefer_new_scene)
        self.scanrefer = scanrefer_new

        # Load camera information captured during annotating
        camera_dict = defaultdict(dict)
        for scene_id in set(unique_scene_ids):
            camera_path = os.path.join(self.root_dir, 'scanrefer', 'annotated_cameras', scene_id + '.anns.json')
            camera_poses = json.load(open(camera_path))
            for cp in camera_poses:
                camera_dict[scene_id][(cp['object_id'], cp['ann_id'])] = cp['camera']
        self.camera = camera_dict

    def _load_data(self):
        '''
        Each mini batch comprises of a single scan, i.e. 3D point cloud coordinates with corresponding
        rgb values, semantic labels and instance labels, and up to self.minibatch_size referrals with
        corresponding descriptions, object and annotation id's, object names and camera positions
        '''
        all_data = []
        for data in tqdm(self.scanrefer, desc=f'Loading {self.split}-split minibatches'):
            minibatch = defaultdict(list)

            scene_id = data[0]['scene_id']
            minibatch['scene_id'] = scene_id

            # Load ScanNet data
            filename_prefix = os.path.join(self.root_dir, 'processed_scans', scene_id)
            minibatch['xyz'] = np.load(filename_prefix+'_xyz.npy')
            minibatch['rgb'] = np.load(filename_prefix+'_rgb.npy')
            minibatch['sem_label'] = np.load(filename_prefix+'_sem_label.npy')
            minibatch['inst_label'] = np.load(filename_prefix+'_inst_label.npy')
            minibatch['superpoint'] = np.load(filename_prefix+'_superpoint.npy')

            # Load ScanRefer data
            for lang_data in data:
                object_id, ann_id = lang_data['object_id'], lang_data['ann_id']
                # Instance ids are be 1-indexed
                minibatch['object_id'].append(int(object_id) + 1)
                minibatch['ann_id'].append(int(ann_id))
                minibatch['description'].append(lang_data['description'])
                minibatch['object_name'].append(lang_data['object_name'])
                minibatch['cam_label'].append(self.camera[scene_id][(object_id, ann_id)])
            all_data.append(minibatch)
        self.data = all_data

    def collate_fn(self, batch):
        inst_bias = 0
        superpoint_bias = 0
        batch_dict = defaultdict(list)
        for i, sample in enumerate(batch):
            scene_id, voxel, xyz, rgb, sem_label, inst_label, superpoint, inst_info, num_inst, \
               object_id, description, object_name, cam_label = sample
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
            object_id[object_id != self.ignore_index] += inst_bias
            batch_dict['lang_label'].append(torch.tensor(object_id).long())
            inst_bias += num_inst

            superpoint += superpoint_bias
            batch_dict['superpoint'].append(torch.from_numpy(superpoint))
            superpoint_bias += (superpoint.max() + 1)

            batch_dict['inst_info'].append(torch.from_numpy(inst_info))
            batch_dict['description'].extend(description)
            batch_dict['object_name'].extend(object_name)
            batch_dict['cam_label'].append(torch.from_numpy(cam_label))
            batch_dict['sentence_batch'].append(len(description))

        voxel = torch.cat(batch_dict['voxel'], 0)
        batch_dict['voxel'] = voxel                                                 # (N,4)
        batch_dict['xyz'] = torch.cat(batch_dict['xyz'], 0).float()                 # (N,3)
        batch_dict['rgb'] = torch.cat(batch_dict['rgb'], 0).float()                 # (N,3)

        batch_dict['sem_label'] = torch.cat(batch_dict['sem_label'], 0).long()      # (N,)
        inst_label = torch.cat(batch_dict['inst_label'], 0).long()
        batch_dict['inst_label'] = inst_label                                       # (N,)
        mask_label = torch.zeros((inst_bias, inst_label.shape[0]), dtype=torch.int)
        for i in range(mask_label.shape[0]):
            mask_label[i] = (inst_label == i).int()
        batch_dict['mask_label'] = mask_label                                       # (I,N)
        batch_dict['inst_info'] = torch.cat(batch_dict['inst_info'], 0).float()
        batch_dict['superpoint'] = torch.cat(batch_dict['superpoint'], 0).long()    # (N,)

        spatial_shape = np.clip((voxel.max(0)[0][1:] + 1).numpy(), self.min_spatial_size, None)
        batch_dict['spatial_shape'] = spatial_shape
        voxel_hash, p2v_map, v2p_map = voxelization_idx(voxel, len(batch), 4)       # 4 = mean
        batch_dict.update({'voxel_hash': voxel_hash, 'p2v_map': p2v_map, 'v2p_map': v2p_map})

        batch_dict['lang_label'] = torch.cat(batch_dict['lang_label'], dim=0)
        batch_dict['cam_label'] = torch.cat(batch_dict['cam_label'], dim=0).float()
        sentence_batch = torch.tensor(batch_dict['sentence_batch'])
        batch_dict['sentence_batch'] = torch.repeat_interleave(torch.arange(sentence_batch.shape[0]), sentence_batch)

        batch_dict['batch_size'] = len(batch)
        return batch_dict


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    root_dir = input()
    ds_train = ScanRefer('train', root_dir)
    train_batch = next(iter(DataLoader(ds_train, batch_size=4, shuffle=True, collate_fn=ds_train.collate_fn)))
    ds_val = ScanRefer('val', root_dir)
    val_batch = next(iter(DataLoader(ds_val, batch_size=4, shuffle=True, collate_fn=ds_val.collate_fn)))
