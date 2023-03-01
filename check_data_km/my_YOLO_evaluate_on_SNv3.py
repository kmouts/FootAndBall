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

from data.SNv3_dataloader import create_snv3_dataset
from data.SoccerNet.utils import vis_gt_pred
from data.augmentation import PLAYER_LABEL, BALL_LABEL, tensor2image

from data.data_reader import my_collate
from data.issia_utils import _ball_detection_stats, ball_boxes_to_centers_list
from ultralytics import YOLO

sys.path.append('..')
print(torch.cuda.get_device_name(0))
torch.cuda.init()

my_device = 'cuda'
snv3_dataset_path = '/mnt/DATA/DATASETS/SOCCERNETv3/SNV3/SNV3_PIP_data_final'
snv3_tmp = '/mnt/DATA/DATASETS/SOCCERNETv3/tmp/'
sanity_file_paths = []

only_ball_frames = False
MAX_PLAYER_HEIGHT = 250  # 0

test_snv3_dataset = create_snv3_dataset(snv3_dataset_path, tmode='test', only_ball_frames=only_ball_frames,
                                        max_player_height=MAX_PLAYER_HEIGHT)
dataloaders = {'test': DataLoader(test_snv3_dataset, batch_size=1,  # batch = 1, for different sizes
                                  num_workers=2,
                                  pin_memory=True, collate_fn=my_collate)}

print('Test set: Dataset size (in batches): {}'.format(len(dataloaders['test'])))

# Load a model
# model = YOLO("yolov8n.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model
# model = YOLO("/mnt/DATA/DFVA/FootAndBall/FootAndBall/runs/detect/yolo8n_custom_640/weights/best.pt")
model = YOLO("/mnt/DATA/DFVA/FootAndBall/FootAndBall/runs/detect/yolo8n_custom_1280_SGD4/weights/best.pt")

phase = 'test'

# https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html
metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True, compute_on_cpu=True,
                              iou_thresholds=[0.5], rec_thresholds=list(np.linspace(0, 1, 11))
                              ).to(torch.device("cpu"))
metric2 = MeanAveragePrecision(iou_type="bbox", class_metrics=True, compute_on_cpu=True).to(torch.device("cpu"))

count_batches = 0
# ball_pos contains list of ball positions (x,y) on each frame; multiple balls per frame are possible
gt_ball_pos = []
pred_ball_pos = []

dtimes = [None] * len(dataloaders['test'])
tdtimes = [None] * len(dataloaders['test'])

# Iterate over data.
for ndx, im_data in enumerate(tqdm(dataloaders[phase])):
    images, boxes, labels, fpaths = im_data[0], im_data[1], im_data[2], im_data[3]
    target = [{'boxes': b.detach().cpu(), 'labels': l.detach().cpu()} for b, l in zip(boxes, labels)]
    # images = images.to(my_device)

    count_batches += 1

    img = tensor2image(images.squeeze(), snv3=True)
    image = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # cv2.imshow(str(ndx), image, dtype = np.uint8)
    # cv2.waitKey()

    # for each image in the batch
    for t in target:
        ball_boxes_to_centers_list(t, gt_ball_pos)

    # results = model(image)

    results = model.predict(source=image, save=False, conf=0.25, show=False, verbose=True,
                            stream=False, device=0, augment=False)

    pred_cpu = {'boxes': [], 'labels': [], 'scores': []}
    pboxes = results[0].boxes.xyxy.detach().cpu()
    plabels = results[0].boxes.cls.detach().cpu()
    pscores = results[0].boxes.conf.detach().cpu()

    dtimes[ndx] = results[0].speed['inference']
    tdtimes[ndx] = results[0].speed['preprocess']+results[0].speed['inference']+results[0].speed['postprocess']

    player_mask = plabels == 1  # COCO (+1) 0 for person, 32 for sports ball
    ball_mask = plabels == 0   # For custom trained YOLO: 0 for BALL, 1 for person
    mask = torch.logical_or(player_mask, ball_mask)

    plabels[player_mask] = PLAYER_LABEL
    plabels[ball_mask] = BALL_LABEL
    plabels = plabels[mask]
    pscores = pscores[mask]
    pboxes = pboxes[mask]

    pred_cpu['boxes'] = pboxes
    pred_cpu['labels'] = plabels
    pred_cpu['scores'] = pscores

    # Update metric with predictions and respective ground truth
    metric.update([pred_cpu], target)
    metric2.update([pred_cpu], target)

    torch.cuda.empty_cache()

    ball_boxes_to_centers_list(pred_cpu, pred_ball_pos)

    # Visualize gt and predictions
    if count_batches == 1 or count_batches % 500 == 0:
        sanity_file_paths.append((count_batches, fpaths[0]))
        # GT bounding boxes
        vis_gt_pred(img, boxes[0], labels[0], pred_cpu, tmp_path=snv3_tmp, batch_num=count_batches,
                    )

pprint(sanity_file_paths)
print("Inference time per image: %3.1f ms" % (sum(dtimes)/len(dataloaders['test'])))
print("Total Inference per image: %3.1f ms" % (sum(tdtimes)/len(dataloaders['test'])))

# Compute the results
result = metric.compute()
pprint(result)
print(result.map.numpy())
print("---------------------------------------------------")
# Compute the results2
result2 = metric2.compute()
pprint(result2)
print(result2.map.numpy())

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
