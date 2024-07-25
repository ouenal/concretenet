# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

'''
Modified from ScanNet evaluation script: https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
and the DKNet adaptation and utils https://github.com/W1zheng/DKNet/blob/main/utils/eval.py
'''
import os, sys
import json
import numpy as np
import torch

CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
OVERLAPS             = np.append(np.arange(0.5,0.95,0.05), 0.25)
MIN_REGION_SIZES     = np.array( [ 100 ] )
DISTANCE_THRESHES    = np.array( [  float('inf') ] )
DISTANCE_CONFS       = np.array( [ -float('inf') ] )
LINELEN = 40


def load_ids(filename):
    ids = open(filename).read().splitlines()
    ids = np.array(ids, dtype=np.int64)
    return ids


class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if (instance_id == -1):
            return
        self.instance_id = int(instance_id)
        self.label_id    = int(self.get_label_id(instance_id))
        self.vert_count  = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"]    = self.label_id
        dict["vert_count"]  = self.vert_count
        dict["med_dist"]    = self.med_dist
        dict["dist_conf"]   = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id     = int(data["instance_id"])
        self.label_id        = int(data["label_id"])
        self.vert_count      = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist    = float(data["med_dist"])
            self.dist_conf   = float(data["dist_conf"])

    def __str__(self):
        return "("+str(self.instance_id)+")"


def get_instances(ids, class_ids, class_labels, id2label):
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(ids)
    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(ids, id)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances


def assign_instances_for_scan(scene_name, pred_info, gt_ids):
    gt_instances = get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)

    gt2pred = gt_instances.copy()
    i = 0
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []
            i+=1
    pred2gt = {}
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0
    bool_void = np.logical_not(np.in1d(gt_ids//1000, VALID_CLASS_IDS))

    nMask = pred_info['label_id'].shape[0]
    for i in range(nMask):
        label_id = int(pred_info['label_id'][i])
        conf = pred_info['conf'][i]
        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        pred_mask = pred_info['mask'][i]
        if len(pred_mask) != len(gt_ids):
            print('wrong number of lines in mask#%d: ' % (i)  + '(%d) vs #mesh vertices (%d)' % (len(pred_mask), len(gt_ids)))
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < MIN_REGION_SIZES[0]:
            continue

        pred_instance = {}
        pred_instance['filename'] = '{}_{:03d}'.format(scene_name, num_pred_instances)
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        matched_gt = []
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy['intersection']   = intersection
                pred_copy['intersection'] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)
    return gt2pred, pred2gt


def evaluate_matches(matches):
    overlaps = OVERLAPS
    min_region_sizes = [MIN_REGION_SIZES[0]]
    dist_threshes = [DISTANCE_THRESHES[0]]
    dist_confs = [DISTANCE_CONFS[0]]

    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), np.float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    gt_instances = [gt for gt in gt_instances if
                                    gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=np.bool)
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt['matched_pred'])
                        for pred in gt['matched_pred']:
                            if pred_visited[pred['filename']]:
                                continue
                            overlap = float(pred['intersection']) / (
                            gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        if not found_match:
                            hard_false_negatives += 1
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (
                            gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                if gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt['dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                if has_gt and has_pred:
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    num_examples = len(y_score_sorted)
                    if(len(y_true_sorted_cumsum) == 0):
                        num_true_examples = 0
                    else:
                        num_true_examples = y_true_sorted_cumsum[-1]
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    precision[-1] = 1.
                    recall[-1] = 0.

                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)

                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                    ap_current = np.dot(precision, stepWidths)
                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di, li, oi] = ap_current
    return ap


def compute_averages(aps):
    d_inf = 0
    o50   = np.where(np.isclose(OVERLAPS,0.5))
    o25   = np.where(np.isclose(OVERLAPS,0.25))
    oAllBut25  = np.where(np.logical_not(np.isclose(OVERLAPS,0.25)))
    avg_dict = {}
    avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[ d_inf,:,o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[ d_inf,:,o25])
    avg_dict["classes"]  = {}
    for (li,label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name]             = {}
        avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,oAllBut25])
        avg_dict["classes"][label_name]["ap50%"]    = np.average(aps[ d_inf,li,o50])
        avg_dict["classes"][label_name]["ap25%"]    = np.average(aps[ d_inf,li,o25])
    return avg_dict


def print_segmentation_results(ciou, aps):
    sep     = ""
    col1    = ":"

    print("")
    print("#" * LINELEN)
    line  = ""
    line += "{:<15}".format("class" ) + sep + col1
    line += "{:>6}".format("IoU"   ) + sep
    line += "{:>6}".format("AP"    ) + sep
    line += "{:>6}".format("AP@50") + sep
    line += "{:>6}".format("AP@25") + sep
    print(line)
    print("-" * LINELEN)

    for (li,label_name) in enumerate(CLASS_LABELS):
        iou  = ciou[li]
        line  = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>6.1f}".format(iou * 100) + sep
        if aps is None:
            print(line)
            continue
        ap_avg  = aps["classes"][label_name]["ap"]
        ap_50o  = aps["classes"][label_name]["ap50%"]
        ap_25o  = aps["classes"][label_name]["ap25%"]
        line += sep + "{:>6.1f}".format(ap_avg * 100) + sep
        line += sep + "{:>6.1f}".format(ap_50o * 100) + sep
        line += sep + "{:>6.1f}".format(ap_25o * 100) + sep
        print(line)

    print("-"*LINELEN)
    miou = np.nanmean(iou)
    line  = "{:<15}".format("average") + sep + col1
    line += "{:>6.1f}".format(miou * 100)  + sep
    if aps is None:
        print(line)
        print("")
        return

    all_ap_avg  = aps["all_ap"]
    all_ap_50o  = aps["all_ap_50%"]
    all_ap_25o  = aps["all_ap_25%"]
    line += "{:>6.1f}".format(all_ap_avg * 100)  + sep
    line += "{:>6.1f}".format(all_ap_50o * 100)  + sep
    line += "{:>6.1f}".format(all_ap_25o * 100)  + sep
    print(line)


def get_aabb(points):
    # Given as [minx, miny, minz, maxx, maxy, maxz]
    return torch.tensor([*points.min(0)[0], *points.max(0)[0]])


def get_iou(aabb1, aabb2):
    if aabb1 is None or aabb2 is None:
        return 0
    stacked = torch.stack((aabb1, aabb2), dim=1)
    max_of_min = stacked[:3].max(1)[0] # Max of min
    min_of_max = stacked[3:].min(1)[0] # Min of max
    intersection = torch.prod(torch.clip(min_of_max - max_of_min, min=0))
    volume1 = torch.prod(torch.abs(aabb1[3:] - aabb1[:3]))
    volume2 = torch.prod(torch.abs(aabb2[3:] - aabb2[:3]))
    return intersection / (volume1 + volume2 - intersection + 1e-12)


def print_grounding_results(ious):
    sep     = ""
    col1    = ":"

    print("#" * LINELEN)
    line  = ""
    line += " U@25" + sep
    line += "{:>7}".format("U@50") + sep
    line += "{:>7}".format("M@25") + sep
    line += "{:>7}".format("M@50") + sep
    line += "{:>7}".format("O@25") + sep
    line += "{:>7}".format("O@50") + sep
    print(line)
    print("-" * LINELEN)

    unique_ious = torch.tensor(ious[0])
    multiple_ious = torch.tensor(ious[1])
    all_ious = torch.cat((multiple_ious, unique_ious))
    line  = ""
    line += "{:>4.2f}".format(100 * (unique_ious>=0.25).sum()/unique_ious.shape[0])
    line += "{:>7.2f}".format(100 * (unique_ious>=0.5).sum()/unique_ious.shape[0])
    line += "{:>7.2f}".format(100 * (multiple_ious>=0.25).sum()/multiple_ious.shape[0])
    line += "{:>7.2f}".format(100 * (multiple_ious>=0.5).sum()/multiple_ious.shape[0])
    line += "{:>7.2f}".format(100 * (all_ious>=0.25).sum()/all_ious.shape[0])
    line += "{:>7.2f}".format(100 * (all_ious>=0.5).sum()/all_ious.shape[0])
    print(line)
    print("#" * LINELEN)
