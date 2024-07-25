# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
from collections import defaultdict
import numpy as np
import torch
from torchmetrics import ConfusionMatrix

import utils.scannet_utils as utils


class Evaluator:
    def __init__(self, num_classes=20,
                 ignore_index=-100,
                 **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        valid_sem = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        self.valid_sem = np.array(valid_sem)

        self.cm = ConfusionMatrix('multiclass', num_classes=num_classes,
                                  ignore_index=ignore_index).cuda()
        self.matches = defaultdict(dict)
        self.ious = [[], []]

    def reset(self):
        self.cm.reset()
        self.matches = defaultdict(dict)
        self.ious = [[], []]

    def update(self, input_dict, output_dict):
        # Update confusion matrix for semantic segmentation
        sem_label = input_dict['sem_label']
        self.cm.update(output_dict['sem_output'].argmax(1), sem_label)

        if 'mask_pred' not in output_dict:
            return

        # Update matches dictionary for instance segmentation
        scene_id = input_dict['scene_id'][0]
        sem_of_inst_output = output_dict['sem_of_inst_output']
        pred_info = {'conf': output_dict['sem_score'].cpu().numpy(),
                     'label_id': self.valid_sem[sem_of_inst_output.long().cpu().numpy()],
                     'mask': output_dict['mask_pred'].cpu().numpy()}

        inst_label = input_dict['inst_label']
        pan_label = torch.zeros(inst_label.shape[0]).type_as(inst_label)
        num_inst = int(inst_label.max()) + 1
        for i in range(num_inst):
            mask_i = inst_label == i
            sem_class = int(sem_label[mask_i][0])
            if sem_class == self.ignore_index: sem_class = 0
            j = self.valid_sem[sem_class]
            pan_label[mask_i] = j * 1000 + i + 1
        pan_label = pan_label.cpu().numpy()
        gt2pred, pred2gt = utils.assign_instances_for_scan(scene_id, pred_info, pan_label)
        self.matches[scene_id] = {'gt': gt2pred, 'pred': pred2gt}

        if 'lang_output' not in output_dict:
            return

        # Update for visual grounding
        # Convert dense mask into aabb
        xyz = input_dict['xyz']
        valid_idx = output_dict['valid_idx']
        lang_output = output_dict['lang_output'][:,valid_idx]
        referred_mask_pred = output_dict['mask_pred'][lang_output.argmax(1)].bool()
        pred_boxes = []
        for _, mask in enumerate(referred_mask_pred):
            aabb = utils.get_aabb(xyz[mask]) if mask.sum() > 0 else None
            pred_boxes.append(aabb)
        # Convert ground truth masks to aabb
        mask_label = input_dict['mask_label'].bool()
        sem_label = input_dict['sem_label']
        target_boxes = []
        is_multiple = []
        for t in input_dict['lang_label']:
            target_mask = mask_label[t]
            sem_class = int(sem_label[target_mask].unique())
            is_multiple.append(sem_class in sem_label[~target_mask])
            target_boxes.append(utils.get_aabb(xyz[target_mask]))
        # Store IoU for unique/multiple
        _ious = []
        for tb, pb, im in zip(target_boxes, pred_boxes, is_multiple):
            iou = utils.get_iou(tb, pb)
            self.ious[int(im)].append(iou)

    def evaluate(self):
        ciou = self.eval_semantic_segmentation()
        ap_dict = self.eval_instance_segmentation()
        utils.print_segmentation_results(ciou, ap_dict)
        self.eval_visual_grounding()

    def eval_semantic_segmentation(self):
        cm = self.cm.compute().cpu().numpy()
        if cm.sum() == 0: return 0, 0
        tp = np.diag(cm)
        with np.errstate(divide='ignore'):
            ciou = tp / (cm.sum(1) + cm.sum(0) - tp)
        miou = np.nanmean(ciou)
        return ciou

    def eval_instance_segmentation(self):
        if not self.matches:
            return None
        ap = utils.evaluate_matches(self.matches)
        ap_dict = utils.compute_averages(ap)
        return ap_dict
    
    def eval_visual_grounding(self):
        if not self.ious[0]:
            return None
        unique_ious = torch.tensor(self.ious[0])
        multiple_ious = torch.tensor(self.ious[1])
        all_ious = torch.cat((multiple_ious, unique_ious))
        utils.print_grounding_results(self.ious)
