# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

'''
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
and: https://github.com/zlccccc/3DVG-Transformer/blob/main/data/scannet/load_scannet_data.py
Load Scannet scenes with vertices and ground truth labels for semantic and instance segmentations
'''
import os, argparse
import pathlib
import json, csv
from plyfile import PlyData
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
import torch
import open3d as o3d
import segmentator

parser = argparse.ArgumentParser()
parser.add_argument('--scannet_dir', help='data directory containing scans')
parser.add_argument('--output_dir', help='output directory, will be appended to scannet_dir', default='processed_scans')
parser.add_argument('--scannet_metadata', help='path to scannetv2 metadata', default='./meta_data/scannetv2_train.txt')
parser.add_argument('--skip_existing', action='store_true', help='skip if the _xyz.npy file exists ')
parser.add_argument('--multiprocessing', action='store_true', help='multiprocessing enabled')
args = parser.parse_args()

REMAPPER = np.ones(150) * (-100)
for i, x in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    REMAPPER[x] = i


def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping


def read_aggregation(filename):
    object_id_to_segs = {}
    object_id_to_label = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            # Instance ids should be 1-indexed
            object_id = data['segGroups'][i]['objectId'] + 1
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            object_id_to_label[object_id] = label
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, object_id_to_label, label_to_segs


def read_segmentation(filename):
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def read_mesh_vertices_rgb(filename):
    assert os.path.isfile(filename)
    f = PlyData().read(filename)
    points = np.array([list(x) for x in f.elements[0]])
    xyz = np.ascontiguousarray(points[:, :3])
    rgb = np.ascontiguousarray(points[:, 3:6])
    return xyz, rgb


def prepare_superpoint(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
    faces = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
    superpoint = segmentator.segment_mesh(vertices, faces).numpy()
    return superpoint


def load_data(mesh_file, agg_file, seg_file, label_map_file):
    label_map = read_label_mapping(label_map_file, label_from='raw_category', label_to='nyu40id')
    xyz, rgb = read_mesh_vertices_rgb(mesh_file)
    superpoint = prepare_superpoint(mesh_file)

    if not os.path.isfile(agg_file):
        return xyz, rgb, superpoint, None, None

    object_id_to_segs, object_id_to_label, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)

    sem_label = np.zeros(shape=(num_verts), dtype=np.uint32)
    object_id_to_label_id = {}
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            sem_label[verts] = label_id
    sem_label = REMAPPER[sem_label]

    inst_label = np.full(shape=(num_verts), fill_value=-100, dtype=np.int32)
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            if object_id_to_label[object_id] == 'wall' or object_id_to_label[object_id] == 'floor':
                continue
            inst_label[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = sem_label[verts][0]
    return xyz, rgb, superpoint, sem_label, inst_label


def export_data(scan_name):
    mesh_file = os.path.join(args.scannet_dir, 'scans', scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(args.scannet_dir, 'scans', scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join(args.scannet_dir, 'scans', scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    label_file = os.path.join(args.scannet_dir, 'scannetv2-labels.combined.tsv')
    xyz, rgb, superpoint, sem_label, inst_label = load_data(mesh_file, agg_file, seg_file, label_file)

    output_filename_prefix = os.path.join(args.scannet_dir, args.output_dir, scan_name)
    np.save(output_filename_prefix+'_xyz.npy', xyz)
    np.save(output_filename_prefix+'_rgb.npy', rgb)
    np.save(output_filename_prefix+'_superpoint.npy', superpoint)
    if sem_label is not None:
        np.save(output_filename_prefix+'_sem_label.npy', sem_label)
        np.save(output_filename_prefix+'_inst_label.npy', inst_label)


def main():
    pathlib.Path(args.scannet_dir, args.output_dir).mkdir(parents=True, exist_ok=True)
    scan_names = sorted([line.rstrip() for line in open(args.scannet_metadata)])
    
    if args.skip_existing:
        new_scan_names = []
        for scan_name in scan_names:
            if not os.path.join(args.scannet_dir, args.output_dir, scan_name + '_xyz.npy'):
                new_scan_names.append(scan_name)
        scan_names = new_scan_names

    if not args.multiprocessing:
        for scan_name in tqdm(scan_names, total=len(scan_names)):
            export_data(scan_name)
        return

    with Pool(processes=cpu_count()) as p:
        with tqdm(total=len(scan_names)) as pbar:
            for _ in p.imap_unordered(export_data, scan_names):
                pbar.update()


if __name__ == '__main__':
    main()
