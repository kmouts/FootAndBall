import os
import pickle
# os.system('nvidia-smi')
import sys
from pprint import pprint

import cv2
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
from ultralytics.yolo.utils.ops import xyxy2xywhn

from data.SNv3_dataloader import create_snv3_dataset
from data.SoccerNet.utils import vis_gt_pred, vis_gt_yolo
from data.SoccerNet.visualize import torch2cv2
from data.augmentation import PLAYER_LABEL, BALL_LABEL, tensor2image

from data.data_reader import my_collate
from data.issia_utils import _ball_detection_stats, ball_boxes_to_centers_list
from network import footandball
from ultralytics import YOLO

sys.path.append('..')

snv3_dataset_path = '/mnt/DATA/DATASETS/SOCCERNETv3/SNV3/SNV3_PIP_data_final'
snv3_tmp = '/mnt/DATA/DATASETS/SOCCERNETv3/tmp/'
snv3_yolo_path = '/mnt/DATA/DATASETS/SOCCERNETv3/SNV3/YOLO/'
sanity_file_paths = []

only_ball_frames = False
MAX_PLAYER_HEIGHT = 250  # 0


def simple_collate(batch):
    _fpaths = []
    _image = batch[0][0]
    _boxes = batch[0][1]
    _labels = batch[0][2]
    _fpath = batch[0][3]
    return _image, _boxes, _labels, _fpath


for phase in ['train', 'valid', 'test']:

    snv3_dataset = create_snv3_dataset(snv3_dataset_path, tmode=phase, only_ball_frames=only_ball_frames,
                                       max_player_height=MAX_PLAYER_HEIGHT, create_for_yolo=True)
    dataloaders = {phase: DataLoader(snv3_dataset, batch_size=1,
                                     num_workers=2,
                                     pin_memory=True, collate_fn=simple_collate)}

    print('{} set: Dataset size (in batches): {}'.format(phase.upper(), len(dataloaders[phase])))

    apath = snv3_yolo_path + phase + "/" + "images"
    if not os.path.exists(apath):
        os.makedirs(apath)
        print("Created: {}".format(apath))
    apath = snv3_yolo_path + phase + "/" + "labels"
    if not os.path.exists(apath):
        os.makedirs(apath)
        print("Created: {}".format(apath))

    count_batches = 0
    gt_ball_pos = []

    # Iterate over data.
    for ndx, im_data in enumerate(tqdm(dataloaders[phase])):
        image, boxes, labels, fpath = im_data[0], im_data[1], im_data[2], im_data[3]

        xywh_boxes = xyxy2xywhn(boxes, w=image.width, h=image.height)
        labels = labels.numpy().tolist()

        count_batches += 1
        plist = fpath.split('/')[-5:]
        plist[-1] = plist[-1][:-4]  # remove image extention
        del plist[-2]  # delete "v3_frames"
        fn = "_".join(plist).replace(" ", "")

        # image.show()
        image.save(snv3_yolo_path + phase + "/" + "images/" + fn + ".jpg")
        with open(snv3_yolo_path + phase + "/" + "labels/" + fn + ".txt", "w") as f:
            for i, ln in enumerate(xywh_boxes.numpy().tolist()):
                f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(labels[i], *ln))

        # Visualize gt and predictions
        if count_batches == 1 or count_batches % 500 == 0:
            sanity_file_paths.append((count_batches, fpath))
            # GT bounding boxes
            vis_gt_yolo(image, boxes.numpy().astype(int), labels, tmp_path=snv3_tmp, batch_num=count_batches,
                        fname=fn)

    pprint(sanity_file_paths)
