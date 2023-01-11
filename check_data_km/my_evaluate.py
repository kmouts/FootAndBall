import os
import pickle
# os.system('nvidia-smi')
import sys
from pprint import pprint

import cv2
import torch
import tqdm as tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np

from data.data_reader import my_collate
from data.issia_dataset_km import P_IssiaDataset, create_issia_dataset
from network import footandball

sys.path.append('data')
print(torch.cuda.get_device_name(0))
torch.cuda.init()
model_name = 'fb1'
model_weights_path = 'models/model_20201019_1416_final.pth'
ball_confidence_threshold = 0.7
player_confidence_threshold = 0.7
my_device = 'cuda'
issia_video_camera_number = 5
issia_dataset_path = '/mnt/DATA/DATASETS/ISSIA-CNR/issia/'

cameras = [6]
only_ball_frames = False

val_issia_dataset = create_issia_dataset(issia_dataset_path, cameras, mode='val', only_ball_frames=only_ball_frames)
dataloaders = {'val': DataLoader(val_issia_dataset, batch_size=4, num_workers=2,
                                 pin_memory=True, collate_fn=my_collate)}

print('Validation set: Dataset size: {}'.format(len(dataloaders['val'].dataset)))

model = footandball.model_factory(model_name, 'detect', ball_threshold=ball_confidence_threshold,
                                  player_threshold=player_confidence_threshold)
model.print_summary(show_architecture=False)
model = model.to(my_device)

if my_device == 'cpu':
    print('Loading CPU weights...')
    state_dict = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
else:
    print('Loading GPU weights...')
    state_dict = torch.load(model_weights_path)

model.load_state_dict(state_dict)
# Set model to evaluation mode

phase = 'val'
model.eval()

# code from "train_detector"

batch_stats = {'loss': [], 'loss_ball_c': [], 'loss_player_c': [], 'loss_player_l': []}

# https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, compute_on_cpu=True,
                              iou_thresholds=[0.5], rec_thresholds=list(np.linspace(0, 1, 11))
                              ).to(torch.device("cpu"))

count_batches = 0
# Iterate over data.
for ndx, (images, boxes, labels) in enumerate(dataloaders[phase]):
    target = [{'boxes': b.detach().cpu(), 'labels': l.detach().cpu()} for b, l in zip(boxes, labels)]
    images = images.to(my_device)
    # h, w = images.shape[-2], images.shape[-1]
    # gt_maps = model.groundtruth_maps(boxes, labels, (h, w))
    # gt_maps = [e.to(my_device) for e in gt_maps]
    count_batches += 1
    print(count_batches)

    predictions = model(images)

    pred_cpu = [{'boxes': e['boxes'].detach().cpu(), 'labels': e['labels'].detach().cpu(),
                 'scores': e['scores'].detach().cpu()} for e in predictions]

    # Update metric with predictions and respective ground truth
    metric.update(pred_cpu, target)

    torch.cuda.empty_cache()

    # if count_batches == 4:
    #     break

# Compute the results
result = metric.compute()
pprint(result)
print(result.map.numpy())

# https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7
# from time import time
# import multiprocessing as mp
# from contextlib import redirect_stdout
#
# with open('count_workerd.txt', 'w') as f:
#     with redirect_stdout(f):
#         for num_workers in range(2, mp.cpu_count(), 2):
#             train_loader = DataLoader(val_issia_dataset,shuffle=True,num_workers=num_workers,
#                                       batch_size=12,pin_memory=True,collate_fn=my_collate)
#             start = time()
#             for epoch in range(1, 2): # (1,3)
#                 for i, data in enumerate(train_loader, 0):
#                     pass
#             end = time()
#             print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
