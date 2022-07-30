
from glob import glob
import numpy as np
from pathlib import Path

from sklearn.neighbors import KDTree
import _pickle
import pickle
import gzip
import sys
from pandaset import DataSet

sys.path.append('/home/csgrad/mbhosale/Lidar_Segmentation/config')

from config import cfg
from tools import DataProcessing as DP
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


ROOT_PATH = (Path(__file__)  / '..' / '..').resolve()
DATASET_PATH = ROOT_PATH / 'pandaset-devkit' / 'data' / 'PandaSet'
NEW_PATH = DATASET_PATH / 'subsampled'
LABELS_PATH = DATASET_PATH / 'classes.json'
TRAIN_PATH = 'train'
TEST_PATH = 'test'
VAL_PATH = 'valid'

LABELS_AVAILABLE_IN_TEST_SET = True

sub_grid_size = cfg.sub_grid_size

for folder in [TRAIN_PATH, VAL_PATH]:
    (NEW_PATH / folder).mkdir(parents=True, exist_ok=True)
    with_semseg = DataSet(DATASET_PATH / folder).sequences(with_semseg=True)
    for file in (DATASET_PATH / folder).rglob('*.pkl.gz'):
        f_parts = list(file.parts)
        if "lidar" in f_parts:
            seq = f_parts[-3]
        else:
            continue
        
        if seq not in with_semseg:
            continue
        scene = file.stem
        
        
        print(file.name, end=':\t')
        if (NEW_PATH / folder / (seq+"_"+scene+'.pkl.gz')).exists():
            print('Already subsampled.')
            continue

        f_parts.remove('lidar')
        f_parts.insert(-1, 'annotations/semseg')
        label_fname = Path(*f_parts)
        print(f'\n Sequence : {seq}, Scene : {scene}')
        with gzip.open(file, 'rb') as f:
            data = _pickle.load(f)
            data = data.to_numpy()
        with gzip.open(label_fname, 'rb') as f:
            labels = _pickle.load(f)
            labels = labels.to_numpy()

        print('Loaded data of shape : ', (data.shape))

        # For each point cloud, a sub sample of point will be used for the nearest neighbors and training
        sub_npy_file = NEW_PATH / folder / (seq + "_" + scene[:-4] + ".npy")
        xyz = data[:,:3].astype(np.float32)
        colors = data[:,3:4].astype(np.uint8)

        if folder!=TEST_PATH or LABELS_AVAILABLE_IN_TEST_SET:
            labels = labels.astype(np.uint8)
            sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, colors, labels, sub_grid_size)
            sub_colors = sub_colors / 255.0
            np.save(sub_npy_file, np.concatenate((sub_xyz, sub_colors, sub_labels), axis=1).T)

        else:
            sub_xyz, sub_colors = DP.grid_sub_sampling(xyz, colors, None, sub_grid_size)
            sub_colors = sub_colors / 255.0
            np.save(sub_npy_file, np.concatenate((sub_xyz, sub_colors), axis=1).T)

        # The search tree is the KD_tree saved for each point cloud
        search_tree = KDTree(sub_xyz)
        kd_tree_file = NEW_PATH / folder / (seq+"_"+ scene[:-4] + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

            # Projection is the nearest points of the selected grid to each point of the cloud
            proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
            proj_idx = proj_idx.astype(np.int32)
            proj_save = NEW_PATH / folder / (seq+"_"+scene + scene[:-4] + '_proj.pkl')
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_idx, labels], f)
