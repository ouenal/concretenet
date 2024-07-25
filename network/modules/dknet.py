# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import dknet_ops

import utils.network_utils as utils
from network.modules.unet import UBlock
from network.modules.dynamic_filter_layer import DynamicFilterLayer


class DKNet(nn.Module):
    def __init__(self, input_dim=6,        # Input dim (rgb, xyz)
                 dim=32,                   # Base dim for UNet
                 output_dim=16,            # Dim of candidate features
                 repetitions=2,            # Repetitions in UNet
                 num_classes=20,           # Number of classes
                 threshold=0.0,            # Threshold for topk NMS
                 local_threshold=0.5,      # Local threshold for topk NMS
                 invalid_classes=2,        # Invalidate wall and floor classes
                 merge_threshold=0.5,      # Threshold for aggregation
                 max_num_instances=80,     # Clip instance count
                 threshold_num_points=100,
                 threshold_sem_score=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.local_threshold = local_threshold
        self.invalid_classes = invalid_classes
        self.merge_threshold = merge_threshold
        self.max_num_instances = max_num_instances
        # For decoding
        self.threshold_num_points = threshold_num_points
        self.threshold_sem_score = threshold_sem_score

        dim_list = [dim, 2*dim, 3*dim, 4*dim, 5*dim, 6*dim, 7*dim]
        self.conv = spconv.SubMConv3d(input_dim, dim, 3, padding=1, bias=False, indice_key='in1')
        self.unet = UBlock(dim_list, repetitions, indice_key_id=1)
        self.bnrelu = spconv.SparseSequential(nn.BatchNorm1d(dim), nn.ReLU())

        self.semantic_branch = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim, bias=True),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, num_classes, bias=True),
        )
        self.offset_branch = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, 3, bias=True)
        )
        self.heatmap_branch = nn.Sequential(
            nn.Linear(dim+3, dim, bias=True),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim, bias=True),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, 1, bias=True)
        )
        self.kernel_branch = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Conv1d(dim, output_dim, 1)
        )
        self.aggregation_branch = nn.Sequential(
            nn.Conv1d(2*output_dim+3, 2*output_dim+3, 1, bias=False),
            nn.BatchNorm1d(2*output_dim+3), nn.ReLU(),
            nn.Conv1d(2*output_dim+3, 2*output_dim+3, 1, bias=False),
            nn.BatchNorm1d(2*output_dim+3), nn.ReLU(),
            nn.Conv1d(2*output_dim+3, 2*output_dim+3, 1, bias=False),
            nn.BatchNorm1d(2*output_dim+3), nn.ReLU(),
            nn.Conv1d(2*output_dim+3, 1, 1)
        )
        self.mask_branch = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Conv1d(dim, output_dim, 1)
        )

        conv_shape = [16, 1]
        kernel_dim = 0
        output_dim_ = output_dim + 3
        for d in conv_shape:
            kernel_dim += output_dim_ * d + d
            output_dim_ = d
        self.weight_generator = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, 1, bias=False),
            nn.BatchNorm1d(output_dim), nn.ReLU(),
            nn.Conv1d(output_dim, kernel_dim, 1)
        )
        self.dynamic_filter = DynamicFilterLayer(conv_shape)

        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)
    
    def forward(self, input_dict, stage=-1):
        ret = {}
        # Parsing input dictionary
        voxel_hash = input_dict['voxel_hash']
        xyz, rgb = input_dict['xyz'], input_dict['rgb']
        v2p_map, p2v_map = input_dict['v2p_map'], input_dict['p2v_map']
        batch_idx = input_dict['voxel'][:, 0].int()
        spatial_shape = input_dict['spatial_shape']
        batch_size = input_dict['batch_size']

        # Point cloud encoding
        f_point = torch.cat((rgb, xyz), 1).float()
        f_voxel = dknet_ops.voxelization(f_point, v2p_map, 4)
        point_cloud = spconv.SparseConvTensor(f_voxel, voxel_hash.int(), spatial_shape, batch_idx.max()+1)
        point_cloud = self.bnrelu(self.unet(self.conv(point_cloud)))
        f_3d = point_cloud.features[p2v_map.long()] # (N,d)
        sem_output = self.semantic_branch(f_3d)     # (N,num_classes)
        sem_pred = sem_output.max(1)[1].long()      # (N,)
        point_offset = self.offset_branch(f_3d)     # (N,3)

        ret.update({'sem_output': sem_output, 'point_offset': point_offset, 'batch_idx': batch_idx})
        if stage == 0: return ret

        # Filter based on semantic class
        object_idx = torch.nonzero(sem_pred >= self.invalid_classes).view(-1)
        xyz_ = xyz[object_idx]
        point_offset_ = point_offset[object_idx]
        batch_idx_ = batch_idx[object_idx]
        batch_offsets_ = utils.get_batch_offsets(batch_idx_, point_cloud.batch_size)
        sem_pred_ = sem_pred[object_idx].int()

        '''
        Candidate generation: heatmap estimation
        center_scores: (N',) float
        candidate_points: (N',) int, point idx of candidates in valid points
        k_foregrounds: (M,2) int, dim 0 for candidate_id, dim 1 for corresponding foreground point idx in valid points
        k_backgrounds: (M',2) int, dim 0 for candidate_id, dim 1 for corresponding background point idx in valid points
        sizes: (N',) int, candidate size, the number of neighboor points
        '''
        f_joint = torch.cat((f_3d[object_idx], point_offset_), dim=1)
        candidate_score = self.heatmap_branch(f_joint)
        candidate_score = utils.batch_softmax(candidate_score, batch_idx[object_idx], dim=0)
        instance_buffer = point_cloud.batch_size*150
        center_scores, candidate_points, k_foregrounds, k_backgrounds, candidate_batch, sizes = \
            utils.topk_nms(candidate_score.detach(), batch_offsets_, batch_idx_, xyz_,
            sem_pred_, 0.3, self.threshold, self.local_threshold, instance_buffer)

        # Candidate aggregation
        f_3d = torch.unsqueeze(f_3d, dim=2).permute(2,1,0)
        num_instances = k_foregrounds[-1, 0] + 1
        f_kernel = self.kernel_branch(f_3d).permute(2,1,0).squeeze()[object_idx]        # (N',d'): float
        f_foreground = utils.neighbor_pooling(f_kernel, k_foregrounds, num_instances)   # (N',d'): float
        f_background = utils.neighbor_pooling(f_kernel, k_backgrounds, num_instances)   # (N',d'): float

        batch_mask = candidate_batch.unsqueeze(1) - candidate_batch.unsqueeze(0)        # (I*I)
        f_candidate = torch.cat((f_foreground, f_background, (xyz_ + point_offset_)[candidate_points]), dim=1)
        d_candidate = torch.clamp(torch.abs(f_candidate.unsqueeze(1) - f_candidate.unsqueeze(0)), min=1e-6)
        merge_score = self.aggregation_branch(d_candidate.permute(0,2,1)).permute(0,2,1).sigmoid()
        merge_score = torch.where(batch_mask.unsqueeze(-1)!=0, torch.zeros_like(merge_score).cuda(), merge_score)
        inst_idx = torch.arange(f_foreground.shape[0])
        merge_score_ = merge_score.clone()

        if stage != 1:
            num_instances = inst_idx.shape[0]
            merge_score_[torch.eye(num_instances).bool()] = 0
            while merge_score_.max() > self.merge_threshold:
                index = merge_score_.argmax()                                                # max score
                i, j = divmod(int(index), num_instances)                                     # candidate i, candidate j
                i_group = torch.where(inst_idx[:] == inst_idx[i])[0]                         # group i, candidates with the same instance_id as candidate i
                j_group = torch.where(inst_idx[:] == inst_idx[j])[0]                         # group j, candidates with the same instance_id as candidate j
                new_group = torch.cat((i_group, j_group), dim=0)                             # merged group
                new_group_h = new_group.view(-1,1).repeat(1,new_group.shape[0])
                new_group_v = new_group.view(1,-1).repeat(new_group.shape[0],1)
                merge_score_[new_group_h, new_group_v] = 0                                   # set scores within the new group to 0

                inst_idx_ = inst_idx.clone()
                inst_idx[new_group] = min(inst_idx_[i], inst_idx_[j])                        # update inst_idx

        merged_idx = torch.unique(inst_idx)                                                  # (I): int
        num_merged_instances = merged_idx.shape[0]
        candidate_points_ = torch.zeros(num_merged_instances).cuda().long()                  # (I): int, point idx of instance center in valid points
        center_scores_ = torch.zeros(num_merged_instances).type_as(center_scores)            # (I): int, point idx of instance center in valid points
        f_kernel = torch.zeros((num_merged_instances, f_foreground.shape[1])).cuda().float() # (I,d'): float, instance kernel
        candidate_batch_ = torch.zeros((num_merged_instances)).cuda().long()
        for i in range(num_merged_instances):
            group = torch.where(inst_idx == merged_idx[i])[0]                                # candidates in the same instance
            center = group[candidate_score[candidate_points[group]].argmax()]                # candidate with the highest center score is the instance center
            candidate_points_[i] = candidate_points[center]
            center_scores_[i] = center_scores[center]
            f_kernel[i] = utils.weighted_mean(sizes[group], f_foreground[group])
            candidate_batch_[i] = candidate_batch[center]

        # Limit the maximum number of allowed instances
        if stage > 0 and num_merged_instances > self.max_num_instances:
            candidate_points_ = candidate_points_[:self.max_num_instances]
            f_kernel = f_kernel[:self.max_num_instances]
            candidate_batch_ = candidate_batch_[:self.max_num_instances]
            center_scores_ = center_scores_[:self.max_num_instances]

        ret.update({'candidate_score': candidate_score.squeeze(), 'candidate_idx': object_idx[candidate_points],
                    'merge_score': merge_score,  'merge_idx': object_idx[candidate_points_],
                    'num_merges': (torch.unique(candidate_batch, return_counts=True)[1]**2).sum()})
        if stage == 1: return ret

        # Mask generation
        f_masks = self.mask_branch(f_3d).permute(2,1,0).squeeze()                                                        # (N,d'): float
        weights = self.weight_generator(torch.unsqueeze(f_kernel, dim=-1).permute(2,1,0)).permute(2,1,0).squeeze(dim=-1) # (I,W), W is up to kernel dim and num: (4x16)->(4x337)
        batch_mask = (batch_idx.repeat(point_cloud.batch_size, 1) == torch.arange(point_cloud.batch_size).cuda().unsqueeze(-1))

        num_points = xyz.shape[0] # N
        candidate_centers = xyz_[candidate_points_]
        conv_masks = batch_mask[candidate_batch_]
        if stage > 0:
            mask_output = torch.zeros((len(weights), xyz.shape[0]), device=f_masks.device)
            for cluster in range(len(weights)):
                position_embedding = xyz - candidate_centers[cluster]
                f_points = torch.cat((f_masks, position_embedding), 1)
                mask_output[cluster, conv_masks[cluster]] = self.dynamic_filter(
                    (f_points[conv_masks[cluster]], weights[cluster])).view(-1)
        else:
            position_embedding = xyz.unsqueeze(0) - candidate_centers.unsqueeze(1)
            f_masks = torch.cat((f_masks.unsqueeze(0).repeat(len(weights),1,1), position_embedding), 2)
            mask_output = self.dynamic_filter((f_masks, weights))
        mask_output = mask_output.view(-1, xyz.shape[0])

        threshold = dknet_ops.otsu(mask_output, 100)
        mask_thresholded = torch.where(mask_output < threshold.unsqueeze(-1),
                                torch.zeros_like(mask_output).cuda(), mask_output)

        inst_score, inst_output = mask_thresholded.max(0)
        inst_output[inst_score < threshold[threshold != 0].min()] = -100
        sem_of_inst_output = utils.get_semantic_output(sem_pred_, inst_output[object_idx],
                                mask_output.shape[0], self.num_classes)
        ret.update({'mask_output': mask_output, 'threshold': threshold, 'mask_thresholded': mask_thresholded,
                    'inst_output': inst_output, 'sem_of_inst_output': sem_of_inst_output, 'f_kernel': f_kernel,
                    'candidate_centers': candidate_centers, 'candidate_batch': candidate_batch_})
        # Pass superpoint for decoding
        ret['superpoint'] = input_dict['superpoint']
        return ret

    def decode_output(self, output_dict):
        # Load relevant keys
        sem_output = output_dict['sem_output']
        mask_output = output_dict['mask_output']
        threshold = output_dict['threshold']
        mask_thresholded = output_dict['mask_thresholded']
        inst_output = output_dict['inst_output']
        sem_of_inst_output = output_dict['sem_of_inst_output']
        merge_idx = output_dict['merge_idx']
        superpoint = output_dict['superpoint']

        ret = {'sem_output': sem_output}
        if 'mask_output' not in output_dict:
            return ret

        # Reevaluate thresholded mask output
        num_inst = mask_output.shape[0]
        mask_score = torch.zeros(num_inst).to(mask_output.device).long()
        for i in range(num_inst):
            mask_score[i] = mask_output[i, inst_output == i].sum()
        mask_score = torch.clamp((mask_score.float() / (mask_thresholded.sum(1) + 1e-6)), max=1.0)
        mask_thresholded = torch.sqrt(mask_thresholded * mask_score.unsqueeze(-1))
        threshold = torch.sqrt(threshold * mask_score)
        inst_score, inst_output = mask_thresholded.max(0)
        inst_output[inst_score < threshold[threshold != 0].min()] = -100

        # Remove predictions with less than threshold_num_points
        num_points_per_mask = torch.zeros(num_inst).cuda().float()
        for i in range(num_inst):
            num_points_per_mask[i] = (inst_output == i).sum()
        valid_inst = num_points_per_mask > self.threshold_num_points
        valid_idx = torch.arange(mask_thresholded.shape[0])[valid_inst]
        merge_idx = merge_idx[valid_inst]
        sem_of_inst_output = sem_of_inst_output[valid_inst]
        mask_score = mask_score[valid_inst]
        threshold = threshold[valid_inst]
        mask_thresholded = mask_thresholded[valid_inst]

        inst_score, inst_output = mask_thresholded.max(0)
        inst_output[inst_score < threshold[threshold != 0].min()] = -100

        # Update semantic output and remove masks with lower semantic confidence
        sem_output_softmax = sem_output.softmax(-1)
        sem_score = torch.zeros_like(mask_score)
        for i in range(valid_inst.sum()):
            if sem_of_inst_output[i] == self.num_classes:
                continue
            num_points_in_mask = (inst_output == i).sum()
            sem_score[i] = mask_thresholded[i, inst_output == i].sum() / (num_points_in_mask + 1e-6)**2 * \
                sem_output_softmax[(inst_output == i), sem_of_inst_output[i].long()].sum()
        sem_score = torch.sqrt(sem_score)

        valid_sem = sem_score > self.threshold_sem_score
        valid_idx = valid_idx[valid_sem]
        merge_idx = merge_idx[valid_sem]
        sem_score = sem_score[valid_sem]
        threshold = threshold[valid_sem]
        mask_thresholded = mask_thresholded[valid_sem]

        inst_score, inst_output = mask_thresholded.max(0)
        inst_output[inst_score < threshold[threshold != 0].min()] = -100

        # Refine instance results with superpoints
        superpoint = torch.unique(superpoint, return_inverse=True)[1]
        superpoint_output, _ = utils.align_superpoint_label(inst_output, superpoint, mask_thresholded.shape[0])
        inst_output = superpoint_output[superpoint]

        # Update semantic classes of predicted instances
        sem_pred = sem_output.max(1)[1]
        object_idx = torch.nonzero(sem_pred >= self.invalid_classes).view(-1)
        sem_of_inst_output = utils.get_semantic_output(sem_pred[object_idx], inst_output[object_idx],
                                mask_thresholded.shape[0], self.num_classes)

        # Extract masks from output
        mask_pred = torch.zeros_like(mask_thresholded).int()
        for i in range(mask_pred.shape[0]):
            mask_pred[i] = (inst_output == i).int()

        # Final filter based on sem output
        valid_sem = sem_of_inst_output != self.num_classes
        valid_idx = valid_idx[valid_sem]
        sem_score = sem_score[valid_sem]
        mask_pred = mask_pred[valid_sem]
        sem_of_inst_output = sem_of_inst_output[valid_sem]

        ret.update({'mask_pred': mask_pred, 'sem_score': sem_score,
                    'sem_of_inst_output': sem_of_inst_output, 'valid_idx': valid_idx})
        return ret
