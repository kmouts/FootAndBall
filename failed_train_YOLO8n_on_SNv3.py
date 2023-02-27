import torchvision
from matplotlib import pyplot as plt
#
# Train FootAndBall detector on ISSIA-CNR Soccer and SoccerPlayerDetection dataset
#

# import tqdm
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import os
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo.utils import RANK
from ultralytics.yolo.utils.ops import xyxy2xywh, xyxy2xywhn

from data.SNv3_dataloader import create_snv3_dataset
from data.augmentation import tensor2image
from network import footandball
from data.data_reader import make_dataloaders, my_collate
from network.ssd_loss import SSDLoss
from misc.config import Params
import random

from ultralytics.yolo.v8.detect import DetectionTrainer

torch.manual_seed(10)
random.seed(10)
np.random.seed(10)
torch.backends.cudnn.benchmark = False

writer = SummaryWriter('runs/fb_exp_train_maxHeight250_200_epochs_deterministic')

# snv3_dataset_path = '/mnt/DATA/DATASETS/SOCCERNETv3/SNV3/SNV3_PIP_data_final'
snv3_dataset_path = '/home/kmouts/Projects/SNV3/SNV3_PIP_data_final'

MODEL_FOLDER = 'models'


def train(params: Params):
    only_ball_frames = False
    MAX_PLAYER_HEIGHT = 250
    val_snv3_dataset = create_snv3_dataset(snv3_dataset_path, tmode='valid', only_ball_frames=only_ball_frames,
                                           max_player_height=MAX_PLAYER_HEIGHT)
    train_snv3_dataset = create_snv3_dataset(snv3_dataset_path, tmode='train', only_ball_frames=only_ball_frames,
                                             max_player_height=MAX_PLAYER_HEIGHT)
    dataloaders = {'val': DataLoader(val_snv3_dataset, batch_size=1, num_workers=params.num_workers,
                                     pin_memory=True, collate_fn=my_collate),
                   'train': DataLoader(train_snv3_dataset, batch_size=params.batch_size, num_workers=params.num_workers,
                                       pin_memory=True, collate_fn=my_collate)}

    print('Training set: Dataset size: {}'.format(len(dataloaders['train'].dataset)))
    if 'val' in dataloaders:
        print('Validation set: Dataset size: {}'.format(len(dataloaders['val'].dataset)))

    class CustomTrainer(DetectionTrainer):
        def get_dataloader(self, dataset_path=snv3_dataset_path, batch_size=params.batch_size, mode="train",
                           rank=0):
            return dataloaders[mode]

        def get_dataset(self, data):
            return train_snv3_dataset, val_snv3_dataset

        def set_model_attributes(self):
            # nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
            # self.args.box *= 3 / nl  # scale to layers
            # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
            # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
            self.data = {"nc": 2, "names": ("BALL", "PLAYER")}
            self.model.nc = self.data["nc"]  # attach number of classes to model
            self.model.names = self.data["names"]  # attach class names to model
            self.model.args = self.args  # attach hyperparameters to model

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = DetectionModel(cfg, ch=3, nc=2, verbose=verbose and RANK == -1)
            if weights:
                model.load(weights)

            return model

        def preprocess_batch(self, batch):
            # my_batch = {"img": batch[0].to(self.device, non_blocking=True).float()}
            # my_batch["bboxes": batch[1].to(self.device, non_blocking=True)]
            # my_batch["cls": batch[2].to(self.device, non_blocking=True)]
            # my_batch = {"img": batch[0], "bboxes": batch[1], "cls": batch[2]}
            im = batch[0].to(self.device, non_blocking=True).float()
            # bb_list = [bboxes.to(self.device) for bboxes in batch[1]]
            # cl_list = [cls.to(self.device) for cls in batch[2]]
            # idx = torch.arange(len(batch[2])).to(self.device)

            # bb_t = xyxy2xywh(torch.cat(batch[1]))
            bb_t = xyxy2xywhn(torch.cat(batch[1]), w=im.shape[-2], h=im.shape[-1])
            bb_t = bb_t.to(self.device, non_blocking=True)
            cl_t = torch.cat(batch[2]).to(self.device, non_blocking=True)

            nt = []
            for i, c in enumerate(batch[2]):  # list of labels
                atemp = torch.full((len(c),), i)
                nt.append(atemp)
            atemp = torch.cat(nt, 0).to(self.device, non_blocking=True)

            my_batch = {"img": im, "bboxes": bb_t, "cls": cl_t, "batch_idx": atemp}
            return my_batch

    trainer = CustomTrainer(overrides={"data": train_snv3_dataset.data, "imgsz": 640,
                                       "batch": params.batch_size, "workers": params.num_workers,
                                       "pretrained": True, "epochs": 20, "patience": 5,
                                       "model": "yolov8n.pt",
                                       "val": True, "device": 0, "name": "yolo8n_custom",
                                       "rect": True, "verbose": False
                                       })
    trainer.train()
    trained_model = trainer.best


if __name__ == '__main__':
    print('Train FoootAndBall detector on SoccerNet v3 dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='configOnSNv3.txt')
    parser.add_argument('--debug', dest='debug', help='debug mode', action='store_true')
    args = parser.parse_args()

    print('Config path: {}'.format(args.config))
    print('Debug mode: {}'.format(args.debug))

    params = Params(args.config, snv3=True)
    params.print()

    train(params)
