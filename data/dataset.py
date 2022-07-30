from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
import pandaset
import math
import gc
import sys
sys.path.append('/home/csgrad/mbhosale/Lidar_Segmentation/config')
from config import cfg
import random


class PandaDataset(Dataset):
    def __init__(self, root_dir, num_scenes, transform=None, to_tensor=False):
        assert isinstance(root_dir, object)
        self.root_dir = root_dir
        self.dataset = pandaset.DataSet(self.root_dir)

        # Only get the datasets with semantic segmentation annotations
        self.sequences = self.dataset.sequences(with_semseg=True)
        self.num_sequences = len(self.sequences)
        self.num_scenes = num_scenes
        self.seq_no = -1
        self.len = self.num_sequences * self.num_scenes
        self.to_tensor = to_tensor
        self.type = type

    def __getitem__(self, idx):
        # using load_lidar to load all data for a scene , then extracting  point cloud and semseg data from it
        # return to_tensor is optional , as we are creating tensors in pandaset_collate

        seq_no = math.floor(idx / self.num_scenes)
        self.scene_no = idx % self.num_scenes
        if self.seq_no != seq_no:
            if self.seq_no != -1:
                del self.lidar
                del self.semseg
                gc.collect()
            self.seq_no = seq_no
            self.seq = self.dataset[self.sequences[self.seq_no]].load_lidar()
            self.seq.load_semseg()
            self.lidar = self.seq.lidar
            self.semseg = self.seq.semseg
        self.sc_semseg = self.semseg.data[self.scene_no]
        self.sc_ptcloud = self.lidar.data[self.scene_no]
        if self.to_tensor:
            self.sc_ptcloud_tensor = torch.tensor(self.sc_ptcloud.values)
            self.sc_semseg_tensor = torch.tensor(self.sc_semseg.values)
            return self.sc_ptcloud_tensor[:, :4], self.sc_semseg_tensor
        else:
            return self.sc_ptcloud.iloc[:, :4], self.sc_semseg

    def __len__(self):
        return self.len



# pandaset_collate is used to return tensor data,
# which are created by  stacking randomly sampled 16000 point cloud and semseg data
def pandaset_collate(batch, args):
    pt_cld = []
    labels = []
    MX_SZ = args['MX_SZ']
    for t in batch:
        idx = random.sample(range(0, t[0].shape[0]), MX_SZ)
        # pt_cld.append(torch.tensor(t[0].iloc[idx].values))
        # labels.append(torch.tensor(t[1].iloc[idx].values))

        l = len(t[0])
        start = 0
        while ((l-start)//MX_SZ) > 0:
            end = MX_SZ + start
            assert((end - start) == MX_SZ)
            pt_cld.append(torch.tensor(t[0].iloc[start:end].values))
            labels.append(torch.tensor(t[1].iloc[start:end].values))
            start = end
    f_pt_cld = torch.stack(pt_cld)
    f_labels = torch.stack(labels)
    return f_pt_cld, f_labels


# passing  pandaset_collate function to data_loader and returning data_loader object
def get_data_loader(dir, batch, MX_SZ, num_scenes=80, to_tensor=True):
    pdset = PandaDataset(root_dir=dir, num_scenes=num_scenes, to_tensor=to_tensor)
    start = 0
    arg = {'MX_SZ': MX_SZ}
    return DataLoader(pdset, batch_size=batch, collate_fn=lambda b: pandaset_collate(b, arg))


# data_loader creates index for input data
if __name__ == '__main__':
    train_dl = get_data_loader(cfg.PATH_TRAIN, 8, 16000, 80, False)
    valid_dl = get_data_loader(cfg.PATH_VALID, 8, 16000, 80, False)
    for i_batch, sample_batched in enumerate(train_dl):
        print(i_batch)
        print(sample_batched)
    #
    # pdset = PandaDataset(r'C:\Users\akumar58\Desktop\instance segmentation\pandaset_0\train', 8)
    # for i in range(len(pdset)):
    #     pt_cloud, label = pdset[i]
