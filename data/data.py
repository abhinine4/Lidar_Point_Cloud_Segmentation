import pickle, time, warnings
import numpy as np
#import logging

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, Sampler, BatchSampler

import sys
sys.path.append('/home/csgrad/mbhosale/Lidar_Segmentation')

from config.config import cfg
from utils.tools import DataProcessing as DP
import pathlib

class PointCloudsDataset(Dataset):
    def __init__(self, dir, labels_available=True):
        self.paths = list(dir.glob(f'*.npy'))
        self.labels_available = labels_available

    def __getitem__(self, idx):
        path = self.paths[idx]

        points, labels = self.load_npy(path)

        points_tensor = torch.from_numpy(points).float()
        labels_tensor = torch.from_numpy(labels).long()

        return points_tensor, labels_tensor

    def __len__(self):
        return len(self.paths)

    def load_npy(self, path):
        r"""
            load the point cloud and labels of the npy file located in path

            Args:
                path: str
                    path of the point cloud
                keep_zeros: bool (optional)
                    keep unclassified points
        """
        cloud_npy = np.load(path, mmap_mode='r').T
        points = cloud_npy[:,:-1] if self.labels_available else points

        if self.labels_available:
            labels = cloud_npy[:,-1]

            # balance training set
            points_list, labels_list = [], []
            for i in range(len(np.unique(labels))):
                try:
                    idx = np.random.choice(len(labels[labels==i]), 8000)
                    points_list.append(points[labels==i][idx])
                    labels_list.append(labels[labels==i][idx])
                except ValueError:
                    continue
            if points_list:
                points = np.stack(points_list)
                labels = np.stack(labels_list)
                labeled = labels>0
                points = points[labeled]
                labels = labels[labeled]

        return points, labels

class CloudsDataset(Dataset):
    def __init__(self, dir, data_type='npy', cloud_split='training'):
        self.path = dir
        self.paths = list(dir.glob(f'*.{data_type}'))
        self.size = len(self.paths)
        self.data_type = data_type
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.val_proj = []
        self.val_labels = []
        self.cloud_split = cloud_split
        # self.val_split = '1_hallway'

        self.load_data()
        print('Size of training : ', len(self.input_colors['training']))
        print('Size of validation : ', len(self.input_colors['validation']))

    def load_data(self):
        for i, file_path in enumerate(self.paths):
            t0 = time.time()
            cloud_name = file_path.stem
            # if self.val_split in cloud_name:
                # cloud_split = 'validation'
            # else:
            # cloud_split = 'training'

            # Name of the input files
            kd_tree_file = self.path / '{:s}_KDTree.pkl'.format(cloud_name)
            sub_npy_file = self.path / '{:s}.npy'.format(cloud_name)

            data = np.load(sub_npy_file, mmap_mode='r').T

            sub_colors = data[:,3:4]
            sub_labels = data[:,-1].copy()

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            # The points information is in tree.data
            self.input_trees[self.cloud_split].append(search_tree)
            self.input_colors[self.cloud_split].append(sub_colors)
            self.input_labels[self.cloud_split].append(sub_labels)
            self.input_names[self.cloud_split].append(cloud_name)

            size = sub_colors.shape[0] * 4 * 7
            print('{} {:s} {:.1f} MB loaded in {:.1f}s'.format(i, kd_tree_file.name, size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Mahesh : Appears it is not being used, so commented as it is giving us the error.
        # # Get validation and test reprojected indices
        # if self.cloud_split == 'validation':
        #     for i, file_path in enumerate(self.paths):
        #         t0 = time.time()
        #         cloud_name = file_path.stem

        #         # Validation projection and labels
        #         if self.val_split in cloud_name:
        #             proj_file = self.path / '{:s}_proj.pkl'.format(cloud_name)
        #             with open(proj_file, 'rb') as f:
        #                 proj_idx, labels = pickle.load(f)

        #             self.val_proj += [proj_idx]
        #             self.val_labels += [labels]
        #             print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds
        return self.size


class ActiveLearningSampler(IterableDataset):

    def __init__(self, dataset, batch_size=6):
        self.dataset = dataset
        # self.split = split
        self.batch_size = batch_size
        self.possibility = {}
        self.min_possibility = {}
        self.split = dataset.cloud_split
        
        if self.split == 'training':
            self.n_samples = cfg.train_steps
        else:
            self.n_samples = cfg.val_steps

        #Random initialisation for weights
        self.possibility[self.split] = []
        self.min_possibility[self.split] = []
        for i, tree in enumerate(self.dataset.input_colors[self.split]):
            self.possibility[self.split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            min_val = self.possibility[self.split][-1]
            self.min_possibility[self.split] += [float(np.min(min_val))]

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.dataset.size # not equal to the actual size of the dataset, but enable nice progress bars

    # Mahesh : Below method tries to firstly randomly select the point cloud and the select randoly the point in that 
    # point cloud (look at the logic of possibility and minpossibility that are used for finding random point cloud and
    # min point in that point cloud respectively). These points are actually the points fromed by the KD tree on the subsampled
    # points of the original point cloud (subsampled points are actually the centre points of the grid in 3d case), these 
    # subsampled points are calculated from the grid sampling done by the GRID subsampling (method = barycenter 
    # for points and features, is this density aware method of subsampling?). Once the random centre point is foound, query
    # is performed on the KD tree of the centre point to get the nearest fg.num_points(currently set to ~80000) number of points. 
    # KDtree is formed in first place beacause it is used to fastly query the nearest points and get the distances, once it is
    # stored in a pickle file (of type sklearn object?) you dont have too rebuild it again and again you could directly use
    # and saves a time than calculating the nearest nabours naively each time. At the end based on the distances the possibilities
    # and min possibilities are updated (increased) for the points which are selected in the current iteration so as to give the 
    # chance to new points in the next iteration (reminder these possibilities and min possibilities are used for random selection of both
    # point cloud and point index in that point cloud.) These approximates to give equal chance to all subsampled points, and
    # grid sampling (subsample.py) handles the sampling based on density.
    def spatially_regular_gen(self):
        # Choosing the least known point as center of a new cloud each time.

        for i in range(self.n_samples * self.batch_size):  # num_per_epoch
            # t0 = time.time()
            if cfg.sampling_type=='active_learning':
                # Generator loop

                # Choose a random cloud
                cloud_idx = int(np.argmin(self.min_possibility[self.split]))

                # choose the point with the minimum of possibility as query point
                point_ind = np.argmin(self.possibility[self.split][cloud_idx])

                # Get points from tree structure
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                if len(points) < cfg.num_points:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    queried_idx = self.dataset.input_trees[self.split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]
                
                # TODO : understand query logic 07/22/2022
                
                queried_idx = DP.shuffle_idx(queried_idx)
                # Collect points and colors
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[self.split][cloud_idx][queried_idx] += delta
                self.min_possibility[self.split][cloud_idx] = float(np.min(self.possibility[self.split][cloud_idx]))

                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

            # Simple random choice of cloud and points in it
            elif cfg.sampling_type=='random':

                cloud_idx = np.random.choice(len(self.min_possibility[self.split]), 1)[0]
                points = np.array(self.dataset.input_trees[self.split][cloud_idx].data, copy=False)
                queried_idx = np.random.choice(len(self.dataset.input_trees[self.split][cloud_idx].data), cfg.num_points)
                queried_pc_xyz = points[queried_idx]
                queried_pc_colors = self.dataset.input_colors[self.split][cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[self.split][cloud_idx][queried_idx]

            queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
            queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
            queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
            queried_idx = torch.from_numpy(queried_idx).float() # keep float here?
            cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()

            points = torch.cat( (queried_pc_xyz, queried_pc_colors), 1)

            yield points, queried_pc_labels


def data_loaders(dir, sampling_method='active_learning', **kwargs):
    if sampling_method == 'active_learning':
        train_dataset = CloudsDataset(dir / 'train', cloud_split='training')
        val_dataset = CloudsDataset(dir / 'valid', cloud_split='validation')
        batch_size = kwargs.get('b_size', 8)
        val_sampler = ActiveLearningSampler(
            val_dataset,
            batch_size=batch_size,
        )
        train_sampler = ActiveLearningSampler(
            train_dataset,
            batch_size=batch_size,
        )
        return DataLoader(train_sampler, **kwargs), DataLoader(val_sampler, **kwargs)

    if sampling_method == 'naive':
        train_dataset = PointCloudsDataset(dir / 'train')
        val_dataset = PointCloudsDataset(dir / 'val')
        return DataLoader(train_dataset, shuffle=True, **kwargs), DataLoader(val_dataset, **kwargs)

    raise ValueError(f"Dataset sampling method '{sampling_method}' does not exist.")

if __name__ == '__main__':
    dataset = CloudsDataset(pathlib.Path('/home/csgrad/mbhosale/Lidar_Segmentation/pandaset-devkit/data/PandaSet/subsampled/train'))
    batch_sampler = ActiveLearningSampler(dataset)
    for data in batch_sampler:
        # xyz, colors, labels, idx, cloud_idx = data
        # print('Number of points:', len(xyz))
        # print('Point position:', xyz[1])
        # print('Color:', colors[1])
        # print('Label:', labels[1])
        # print('Index of point:', idx[1])
        # print('Cloud index:', cloud_idx)
        xyz, labels = data
        print('Number of points:', len(xyz))
        print('Label', labels[1])
        break
