# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import random
import numpy as np
import torch
import dknet_ops
from scipy.sparse import coo_matrix


def get_batch_offsets(batch_idx, bs):
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idx == i).sum()
    assert batch_offsets[-1] == batch_idx.shape[0]
    return batch_offsets


def batch_softmax(input, batch_idx, dim=0):
    batch_size = batch_idx.max() + 1
    for batch in range(batch_size):
        batch_mask = batch_idx == batch
        input[batch_mask] = torch.softmax(input[batch_mask], dim=0)
    return input


def topk_nms(input, batch_offset, batch_idx, coord, label, R=0.3, threshold=0.3, local_threshold=0.5, K=100):
    batch_offset = batch_offset.cpu()
    topk_idx, sizes, k_foregrounds, k_backgrounds = \
        dknet_ops.topk_nms(input, batch_offset, coord, label, R, threshold, local_threshold, K, 2000, 2000)
    topk = input[topk_idx].cuda()
    topk_idx = topk_idx.cuda()
    k_foregrounds = k_foregrounds.long().cuda()
    k_backgrounds = k_backgrounds.long().cuda()
    return topk, topk_idx, k_foregrounds, k_backgrounds, batch_idx[topk_idx], sizes.float().cuda()


def neighbor_pooling(input, masks, num_instances=0):
    if num_instances == 0:
        num_instances = masks[-1, 0] + 1
    input_mean = input.mean(0).detach()
    pooling_output = torch.zeros((num_instances, input.shape[1])).cuda()
    for i in range(num_instances):
        mask = masks[masks[:, 0] == i, 1]
        if mask.shape[0] == 0:
            pooling_output[i] = input_mean
        else:
            mask_input = input[mask]
            maxpooling = mask_input.mean(0)
            pooling_output[i] = maxpooling
    return pooling_output


def weighted_mean(weights, input):
    return (weights.unsqueeze(-1)*input).sum(0) / weights.sum()


def get_semantic_output(semantic_label, seg_result, instance_num=0, num_classes=20):
    # Voting within the predicted instance to obtain semantic labels
    if instance_num == 0:
        instance_num = seg_result.max() + 1
    seg_labels = []
    for n in range(instance_num):
        mask = (seg_result == n)
        if mask.sum() == 0:
            seg_labels.append(num_classes)
            continue
        seg_label_n = torch.mode(semantic_label[mask])[0].item()
        seg_labels.append(seg_label_n)
    return torch.Tensor(seg_labels).cuda()


def mask_augment(referrals, object_name, mask_token, p=0.5):
    masked_ref = []
    for ref, name in zip(referrals, object_name):
        if random.random() < p:
            ref = ref.replace(name.replace('_', ' '), mask_token)
        masked_ref.append(ref)
    return masked_ref


def mean_pooling(f_word, padding_mask):
    padding_mask_expanded = padding_mask.unsqueeze(-1).expand(f_word.size()).float()
    return torch.sum(f_word * padding_mask_expanded, 1) / torch.clamp(padding_mask_expanded.sum(1), min=1e-9)


def align_superpoint_label(labels, superpoint, num_label=20, ignore_label=-100):
    row = superpoint.cpu().numpy()
    col = labels.cpu().numpy()
    col[col < 0] = num_label
    data = np.ones(len(superpoint))
    shape = (len(np.unique(row)), num_label + 1)
    label_map = coo_matrix((data, (row, col)), shape=shape).toarray()
    label = torch.Tensor(np.argmax(label_map, axis=1)).long().to(labels.device)
    label[label == num_label] = ignore_label
    label_scores = torch.Tensor(label_map.max(1) / label_map.sum(axis=1)).to(labels.device)
    return label, label_scores
