import os
import pickle
# os.system('nvidia-smi')
import sys
from collections import defaultdict
from pprint import pprint
import copy

import cv2
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np

from data.SNv3_dataloader import create_snv3_dataset
from data.SoccerNet.visualize import torch2cv2
from data.augmentation import tensor2image
from data.data_reader import my_collate
from data.issia_dataset_km import P_IssiaDataset, create_issia_dataset
from data.issia_utils import evaluate_ball_detection_results, _ball_detection_stats, ball_boxes_to_centers_list
from network import footandball

sys.path.append('..')
print(torch.cuda.get_device_name(0))
torch.cuda.init()
model_name = 'fb1'
model_weights_path = 'models/model_20201019_1416_final.pth'
ball_confidence_threshold = 0.7
player_confidence_threshold = 0.7
my_device = 'cuda'
snv3_dataset_path = '/mnt/DATA/DATASETS/SOCCERNETv3/SNV3/SNV3_PIP_data_final'
snv3_tmp = '/mnt/DATA/DATASETS/SOCCERNETv3/tmp/'

only_ball_frames = False

val_snv3_dataset = create_snv3_dataset(snv3_dataset_path, tmode='valid', only_ball_frames=only_ball_frames)
dataloaders = {'val': DataLoader(val_snv3_dataset, batch_size=4, num_workers=2,
                                 pin_memory=True, collate_fn=my_collate)}

print('Validation set: Dataset size: {}'.format(len(dataloaders['val'])))

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

# https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, compute_on_cpu=True,
                              iou_thresholds=[0.5], rec_thresholds=list(np.linspace(0, 1, 11))
                              ).to(torch.device("cpu"))
metric2 = MeanAveragePrecision(iou_type="bbox", class_metrics=True, compute_on_cpu=True).to(torch.device("cpu"))

count_batches = 0
# ball_pos contains list of ball positions (x,y) on each frame; multiple balls per frame are possible
gt_ball_pos = []
pred_ball_pos = []

# Iterate over data.
for ndx, (images, boxes, labels) in enumerate(tqdm(dataloaders[phase])):
    target = [{'boxes': b.detach().cpu(), 'labels': l.detach().cpu()} for b, l in zip(boxes, labels)]
    images = images.to(my_device)

    count_batches += 1

    # for each image in the batch
    for t in target:
        ball_boxes_to_centers_list(t, gt_ball_pos)

    predictions = model(images)

    pred_cpu = [{'boxes': e['boxes'].detach().cpu(), 'labels': e['labels'].detach().cpu(),
                 'scores': e['scores'].detach().cpu()} for e in predictions]

    # Update metric with predictions and respective ground truth
    metric.update(pred_cpu, target)
    metric2.update(pred_cpu, target)

    torch.cuda.empty_cache()

    # for each image in the batch prediction
    for t in pred_cpu:
        ball_boxes_to_centers_list(t, pred_ball_pos)

    # Visualize gt and predictions
    if count_batches % 100 == 0:
    # Person bounding boxes
    #     frame = copy.deepcopy(images[0]).numpy(force=True)
        frame = tensor2image(images[0]) * 255
        # frame = torch2cv2(images[0].cpu()) * 255
        # frame = frame.transpose(1, 2, 0)
        for (x1, y1, x2, y2), lb in zip((np.rint(boxes[0].numpy(force=True))).astype(int), labels[0].numpy(force=True)):
            color = (0, 0, 0)
            if lb == 1:  # ball
                color = (0, 0, 153)
            elif lb == 2:  # player
                color = (204, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color)
            # cv2.imshow('frame', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
            cv2.imwrite(snv3_tmp + 'batch_' + str(count_batches) + '_0.png', frame)

# Compute the results
result = metric.compute()
pprint(result)
print(result.map.numpy())
print("---------------------------------------------------")
# Compute the results2
result2 = metric2.compute()
pprint(result2)
print(result2.map.numpy())

# https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7

# From issia_utils.py
# Ball detection in pixels performance
tolerance = 5  # 3
frame_stats = []
for i in range(len(pred_ball_pos)):
    ball_pos = pred_ball_pos[i]
    truth_ball_pos = gt_ball_pos[i]
    frame_stats.append(_ball_detection_stats(ball_pos, truth_ball_pos, tolerance))

percent_correctly_classified_frames = sum([c for (_, _, c) in frame_stats]) / len(frame_stats)
temp = [p for (p, _, _) in frame_stats if p is not None]
avg_precision = sum(temp) / len(temp)

temp = [r for (_, r, _) in frame_stats if r is not None]
avg_recall = sum(temp) / len(temp)

print('Avg. ball precision (tolerance: ' + str(tolerance) + ' pixels) = ' + str(avg_precision))
print('Avg. ball recall = ' + str(avg_recall))
print('Percent of correctly classified ball frames = ' + str(percent_correctly_classified_frames))
