import argparse
from datetime import datetime
import numpy as np
from pathlib import Path
import logging
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from models.RandlaNet_mb import *
from utils.tools import Config as cfg
from metrics.metrics import calc_accuracy, calc_iou
from data.data import data_loaders
from utils.utils import AverageMeter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b_size', type=int, help='Batch size for data loader', default=1)
    parser.add_argument('--MX_SZ', type=int, help='max size of point cloud', default=16000)
    parser.add_argument('--n_scenes', type=int, help='number of images per sequence', default=80)
    parser.add_argument('--log_dir', type=str, help='path to log file', default='logs/')
    parser.add_argument('--k', type=int, help='k neighbors', default=16)
    parser.add_argument('--d_in', type=int, help='input feature dimension', default=6)
    parser.add_argument('--decimation', type=int, help='decimation value', default=4)
    # TODO Need to change the number of classes here, just added +1 to avoid the assertion error,
    # https://github.com/scaleapi/pandaset-devkit/issues/132
    parser.add_argument('--num_classes', type=int, help='number of classes in dataset', default=14)
    parser.add_argument('--device', type=str, help='cpu/cuda', default='cuda')
    parser.add_argument('--epochs', type=int, help='number of train epochs', default=20)
    parser.add_argument('--save_freq', type =int, help='save frequency for model', default = 5)
    return parser.parse_args()

def eval(model, pdset_valid, criterion, args):
    model.eval()
    device = args.device
    val_itr_loss = AverageMeter()
    per_class_accs = []
    per_class_ious = []
    with torch.no_grad():
        for idx, data in enumerate(pdset_valid):
            data = (data[0].to(device), data[1].to(device))
            valid_pts, valid_gt_labels = data
            valid_gt_labels = valid_gt_labels.squeeze(-1)
            val_scores = model(valid_pts)
            val_labels = torch.distributions.utils.probs_to_logits(val_scores, is_binary=False)
            val_loss = criterion(val_labels, valid_gt_labels)
            val_itr_loss.update(val_loss.item())
            per_class_accs.append(calc_accuracy(val_labels, valid_gt_labels))
            per_class_ious.append(calc_iou(val_labels, valid_gt_labels))
    return val_itr_loss, per_class_accs, per_class_ious

def randla_train(PATH, args):

    sampling = 'active_learning'
    dataset = Path(PATH)
    pdset_train, pdset_valid = data_loaders(dataset, sampling, batch_size=args.b_size, num_workers=0, pin_memory=True)
    d_in = next(iter(pdset_train))[0].size(-1)
    model = RandLANet(args.k, d_in, args.decimation, args.num_classes, args.device)
    epochs = args.epochs
    log_dir = args.log_dir
    device = args.device
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  #TODO :add class weights for handling class imbalance, #Attention, this could reason 
    # model.double().to(device) Attention, why this is giving error now ?
    num_classes = args.num_classes

    handlers = [logging.StreamHandler()]
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, 'logs_'+str(datetime.now().strftime("%m%d%y_%H%M%S")))
    open(log_file, 'w')
    handlers.append(logging.FileHandler(log_file, mode='a'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)

    logging.info(args)
    tb_logs = os.path.join(log_dir, 'tb')

    with SummaryWriter(tb_logs) as writer:
        model.train()
        for e in range(epochs):
            logging.info(f'===========Epoch {e}/{epochs}============')
            itr_loss = AverageMeter()
            ln = len(pdset_train)
            batch_train_acc = []
            batch_train_iou = []
            for idx, data in enumerate(pdset_train):
                data = (data[0].to(device), data[1].to(device))
                pt_cloud, pt_labels = data

                # torch.save(pt_cloud, 'pts_clpud.pt')
                # torch.save(pt_labels, 'pt_labels.pt')
                pt_labels = pt_labels.squeeze(-1)
                # pt_labels = pt_labels - 1
                # print(pt_labels.max(), pt_labels.min())

                opt.zero_grad()
                scores = model(pt_cloud)

                # Information on logits - https://tinyurl.com/6wp4uwwz
                pred_label = torch.distributions.utils.probs_to_logits(scores, is_binary=False)
                train_loss = criterion(pred_label, pt_labels)
                batch_train_acc.append(calc_accuracy(pred_label, pt_labels))
                batch_train_iou.append(calc_iou(pred_label, pt_labels))

                itr_loss.update(train_loss.item())

                logging.info(f'itreration : {idx}/{ln}\t loss : {train_loss.item()}')
                train_loss.backward()
                opt.step()

            train_accs = np.mean(np.array(batch_train_acc), axis=0)
            train_ious = np.mean(np.array(batch_train_iou), axis=0)

            writer.add_scalar("train/train_loss", itr_loss.avg, e)
            
            ###Evaluation
            eval_loss, eval_accs, eval_ious = eval(model, pdset_valid, criterion, args)
            eval_ious = np.mean(np.array(eval_ious), axis=0)
            eval_accs = np.mean(np.array(eval_accs), axis=0)


            acc_dicts = [ 
                    {   'train_acc' : train_acc,
                        'eval_acc' : eval_acc
                    }for train_acc, eval_acc in zip(train_accs, eval_accs)
            ]

            iou_dicts = [ 
                    {   'train_acc' : train_iou,
                        'eval_acc' : eval_iou
                    }for train_iou, eval_iou in zip(train_ious, eval_ious)
            ]
            

            writer.add_scalar("train/eval_loss", eval_loss.avg, e)
            logging.info(f'Epoch completed : {e}/{epochs} Train_loss : {itr_loss.avg} Validation_loss : {eval_loss.avg} Validation_accuracy : {eval_accs[-1]} Validation_IOU : {eval_ious[-1]}')
            
            # writer syntax : https://pytorch.org/docs/stable/tensorboard.html
            for c in range(len(train_accs)-1):
                writer.add_scalars(f"per-class accuracy/{c:02d}", acc_dicts[c], e)
                writer.add_scalars(f"per-class IoU/{c:02d}", iou_dicts[c], e)
            writer.add_scalars(f"per-class accuracy/overall", acc_dicts[-1], e)
            writer.add_scalars(f"per-class IoU/mean IOU", iou_dicts[-1], e)            

            if e%args.save_freq == 0:
                torch.save(
                    {'epoch' : e,
                    'model_state' : model.state_dict(),
                    'optimizer_state' : opt.state_dict(),
                    'loss_at_epoch' : {
                        'train' : itr_loss.avg,
                        'valid_loss' : eval_loss.avg
                    }},
                    f'{log_dir}/saved_models/model_{e}_{str(datetime.now().strftime("%m%d%y_%H%M%S"))}'
                )
           
if __name__ == "__main__":
    logging.captureWarnings(True)
    args = get_args()
    PATH = '/home/csgrad/mbhosale/RandLA-Net-pytorch/datasets/s3dis/subsampled' 
    randla_train(PATH, args)
