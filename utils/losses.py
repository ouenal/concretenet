# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class JointLoss(nn.Module):
    def __init__(self, visual_config, fusion_config):
        super().__init__()
        self.visual_loss = PanopticLoss(**visual_config)
        self.fusion_loss = DenseGroundingLoss(**fusion_config)
    
    @staticmethod
    def parse_losses(loss_dict):
        loss_dict['loss_total'] = sum(loss_dict.values())
        return loss_dict

    def forward(self, input_dict, output_dict):
        loss = self.visual_loss(input_dict, output_dict)
        if 'lang_output' not in output_dict:
            return self.parse_losses(loss)
        loss.update(self.fusion_loss(input_dict, output_dict))
        return self.parse_losses(loss)


class PanopticLoss(nn.Module):
    def __init__(self, ignore_index=-100,
                 invalid_classes=2,
                 num_classes=20,
                 iou_threshold=0.25):
        super().__init__()
        self.ignore_index = ignore_index
        self.invalid_classes = invalid_classes
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

        self.xe = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.bxe = nn.BCELoss(reduction='none')

    @staticmethod
    def dice_loss(mask_pred, mask_gt, ep=1e-8):
        inter = 2 * (mask_gt * mask_pred).sum(1) + 1
        union = (mask_gt ** 2.0).sum(1) + (mask_pred ** 2.0).sum(1) + 1 + ep
        dice_loss = 1 - inter / union
        return dice_loss

    @staticmethod
    def multi_class_dice_loss(input, target, epsilon=1e-5, weight=None):
        assert input.size() == target.size()
        axis_order = (1, 0) + tuple(range(2, input.dim()))
        input = input.permute(axis_order)
        target = target.permute(axis_order)
        target = target.float()
        per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / \
                        (torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)
        loss = 1. - per_channel_dice
        return loss

    def forward(self, input_dict, output_dict):
        loss_dict = {}

        sem_output = output_dict['sem_output']
        sem_label = input_dict['sem_label']
        loss_sem = self.xe(sem_output, sem_label)

        ignore_mask = sem_label != self.ignore_index
        one_hot_labels = F.one_hot(sem_label[ignore_mask], num_classes=self.num_classes)
        sem_output_softmax = F.softmax(sem_output[ignore_mask], dim=-1)
        loss_sem += self.multi_class_dice_loss(sem_output_softmax, one_hot_labels).mean()
        loss_dict['loss_sem'] = loss_sem

        # Offsets loss
        inst_info = input_dict['inst_info']
        xyz = input_dict['xyz']
        point_offset = output_dict['point_offset']
        inst_label = input_dict['inst_label']
        gt_offset = inst_info[:, 0:3] - xyz                                       # (N,3)
        dist_offset = torch.sum(torch.abs(point_offset - gt_offset), dim=-1)      # (N,)
        valid_inst = (inst_label != self.ignore_index).float()                    # (N,)
        loss_off = torch.sum(dist_offset * valid_inst) / (torch.sum(valid_inst) + 1e-6)

        gt_offset_norm = torch.norm(gt_offset, p=2, dim=1)                        # (N,)
        gt_offset_ = gt_offset / (gt_offset_norm.unsqueeze(-1) + 1e-8)            # (N,3)
        point_offset_norm = torch.norm(point_offset, p=2, dim=1)                  # (N,)
        point_offset_ = point_offset / (point_offset_norm.unsqueeze(-1) + 1e-8)   # (N,3)
        dif_offsets_ = -(gt_offset_ * point_offset_).sum(-1)                      # (N,)
        loss_off += torch.sum(dif_offsets_ * valid_inst) / (torch.sum(valid_inst) + 1e-6)
        loss_dict['loss_off'] = loss_off

        if 'candidate_score' not in output_dict:
            return loss_dict

        # Center heatmap loss
        candidate_score = output_dict['candidate_score']
        sem_pred = sem_output.max(1)[1].long()
        object_idx = torch.nonzero(sem_pred >= self.invalid_classes).view(-1)
        guassian_mean = torch.norm(gt_offset[object_idx], dim=1)
        guassian_std = (inst_info[object_idx, 6:9] - inst_info[object_idx, 3:6]).max(1)[0]
        gt_heatmap = torch.exp(-25*((guassian_mean**2)/((guassian_std+1e-4)**2)))
        valid_joint = valid_inst[object_idx]
        loss_cen = torch.sum(torch.abs(candidate_score - gt_heatmap) * valid_joint) / (torch.sum(valid_joint) + 1e-6)
        loss_dict['loss_cen'] = loss_cen

        # Aggregation loss
        merge_score = output_dict['merge_score']
        mask_label = input_dict['mask_label']
        candidate_idx = output_dict['candidate_idx']
        num_merges = output_dict['num_merges']
        gt_merge = torch.zeros_like(merge_score)
        for i in range(merge_score.shape[0]):
            j = torch.where(mask_label[:, candidate_idx[i]] == 1)[0]
            if j.shape[0] == 0: continue
            gt_merge[i] = mask_label[j, candidate_idx].unsqueeze(-1)
        loss_agg = self.bxe(merge_score, gt_merge).sum() / num_merges
        loss_agg += self.dice_loss(merge_score, gt_merge, 1e-6).mean()
        loss_dict['loss_agg'] = loss_agg

        if 'mask_output' not in output_dict:
            return loss_dict

        # Preparation for mask loss
        point_batch = input_dict['voxel'][:, 0].int()
        batch_size = input_dict['batch_size']
        merge_idx = output_dict['merge_idx']
        sem_of_inst_output = output_dict['sem_of_inst_output']
        batch_mask = (point_batch.repeat(batch_size, 1) == torch.arange(batch_size).cuda().unsqueeze(-1))
        inst_idx = torch.max(mask_label, dim=1)[1].unsqueeze(1)

        # Hungarian algorithm
        # LSA cost = center distance cost + semantic cost + 100 * batch cost
        cost_matrix = torch.norm((inst_info[inst_idx, 0:3] - xyz[merge_idx].unsqueeze(0)), dim=-1)
        cost_matrix += ((sem_label[inst_idx].cuda() - sem_of_inst_output.unsqueeze(0)) != 0).float()
        cost_matrix += 100 * ((point_batch[inst_idx].cuda() - point_batch[merge_idx].unsqueeze(0)) != 0).float()
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu())
        row_ind, col_ind = torch.Tensor(row_ind).cuda().long(), torch.Tensor(col_ind).cuda().long()
        row_ind = row_ind % inst_idx.shape[0]

        # Disallow inter-batch assignments
        cost_mask = (cost_matrix[row_ind, col_ind] < 100)
        if cost_mask.sum():
            row_ind, col_ind = row_ind[cost_mask], col_ind[cost_mask]
        col_ind_sorted, col_ind_order = torch.sort(col_ind)
        row_ind_sorted = row_ind[col_ind_order]

        # Assign ground truth to predictions
        mask_output = output_dict['mask_output']
        mask_output = mask_output[col_ind_sorted]
        merge_idx = merge_idx[col_ind_sorted]
        assigned_inst = torch.index_select(mask_label.float(), 0, row_ind_sorted)
        assigned_sem = sem_label[inst_idx][row_ind_sorted].squeeze()

        # Compute IoUs
        inst_output = output_dict['inst_output']
        inst_output_sorted = torch.zeros_like(mask_output).cuda()
        for i in range(inst_output_sorted.shape[0]):
            inst_output_sorted[i] = (inst_output == col_ind_sorted[i]).int()
        intersection = (inst_output_sorted * assigned_inst).sum(1)
        union = inst_output_sorted.sum(1) + assigned_inst.sum(1) - intersection
        ious = intersection / (union + 1e-6)

        # Filter valid assignments
        valid_assignments = (assigned_sem != self.ignore_index) & (ious > self.iou_threshold)
        if not valid_assignments.sum():
            valid_assignments = (assigned_sem != self.ignore_index)
        mask_output = mask_output[valid_assignments]
        assigned_inst = assigned_inst[valid_assignments]
        merge_idx = merge_idx[valid_assignments]

        # Mask loss
        valid_joint = batch_mask[point_batch[merge_idx].long()].view(-1)
        loss_mask = self.bxe(mask_output.view(-1), assigned_inst.view(-1))[valid_joint].mean()
        loss_mask += self.dice_loss(mask_output, assigned_inst, 1e-6).mean()
        loss_dict['loss_mask'] = loss_mask

        # Save Hungarian algorithm results for the grounding loss
        output_dict.update({'row_ind_sorted': row_ind_sorted, 'col_ind_sorted': col_ind_sorted})
        return loss_dict


class DenseGroundingLoss(nn.Module):
    def __init__(self, ignore_index=-100,
                 contrast_loss_weight=0.1):
        super().__init__()
        self.ignore_index = ignore_index
        self.contrast_loss_weight = contrast_loss_weight
        self.xe = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.mse = nn.MSELoss()

    def forward(self, input_dict, output_dict):
        loss_dict = {}

        # Selection loss
        lang_label = input_dict['lang_label']
        lang_output = output_dict['lang_output']
        mask_label = input_dict['mask_label']
        row_ind_sorted = output_dict['row_ind_sorted']
        col_ind_sorted = output_dict['col_ind_sorted']
        inst_ind_sorted = torch.arange(mask_label.shape[0])[row_ind_sorted].cuda()
        lang_label_sorted = torch.full_like(lang_label, self.ignore_index)
        for i, ind in enumerate(lang_label):
            if ind == self.ignore_index:
                continue
            ind_matching = inst_ind_sorted == ind
            if ind_matching.sum() > 0:
                assert ind_matching.sum() == 1 # This should always be true from LSA
                lang_label_sorted[i] = int(ind_matching.nonzero(as_tuple=True)[0])
        lang_output_sorted = lang_output[:,col_ind_sorted]

        loss_sel = self.xe(lang_output_sorted, lang_label_sorted)
        loss_dict['loss_sel'] = loss_sel

        # Contrastive loss
        f_instance = output_dict['f_instance']
        f_sentence = output_dict['f_sentence']
        sentence_batch = input_dict['sentence_batch']
        candidate_batch = output_dict['candidate_batch']
        valid_lang = lang_label_sorted != self.ignore_index
        lang_label_sorted = lang_label_sorted[valid_lang]
        f_instance = F.normalize(f_instance[valid_lang], p=2, dim=2) # (L,I,C)
        f_sentence = f_sentence[valid_lang]                          # (L,C)
        similarity = torch.exp(F.cosine_similarity(f_sentence[:,None], f_instance, dim=-1) / 0.3)

        diffs = torch.eq(sentence_batch[valid_lang][:,None], candidate_batch[None])
        pos = similarity[torch.arange(lang_label_sorted.shape[0]), lang_label_sorted]
        neg = (similarity * diffs).sum(1) + 1e-8
        loss_con = -torch.log(pos / neg).mean()
        loss_dict['loss_con'] = self.contrast_loss_weight * loss_con

        # Camera loss
        cam_output = output_dict['cam_output']
        cam_label = input_dict['cam_label']
        loss_cam = self.mse(cam_output, cam_label)
        loss_dict['loss_cam'] = loss_cam
        return loss_dict
