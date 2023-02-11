import os
import pickle
# os.system('nvidia-smi')
import sys
from pprint import pprint

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np

from data.SNv3_dataloader import create_snv3_dataset
from data.SoccerNet.utils import vis_gt_pred

from data.data_reader import my_collate
from data.issia_utils import  _ball_detection_stats, ball_boxes_to_centers_list
from network import footandball

sys.path.append('..')
print(torch.cuda.get_device_name(0))
torch.cuda.init()
model_name = 'fb1'
model_weights_path = 'models/model_20201019_1416_final.pth'  # original
# model_weights_path = 'models/model_20230209_1818_final.pth'  # 150epochs deterministic train on whole SNv3
ball_confidence_threshold = 0.7
player_confidence_threshold = 0.7
my_device = 'cuda'
snv3_dataset_path = '/mnt/DATA/DATASETS/SOCCERNETv3/SNV3/SNV3_PIP_data_final'
snv3_tmp = '/mnt/DATA/DATASETS/SOCCERNETv3/tmp/'
sanity_file_paths = []

only_ball_frames = False

test_snv3_dataset = create_snv3_dataset(snv3_dataset_path, tmode='test', only_ball_frames=only_ball_frames)
dataloaders = {'test': DataLoader(test_snv3_dataset, batch_size=1,  # batch = 1, for different sizes
                                  num_workers=2,
                                 pin_memory=True, collate_fn=my_collate)}

print('Test set: Dataset size (in batches): {}'.format(len(dataloaders['test'])))

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

phase = 'test'
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
for ndx, im_data in enumerate(tqdm(dataloaders[phase])):
    images, boxes, labels, fpaths = im_data[0], im_data[1], im_data[2], im_data[3]
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
    if count_batches == 1 or count_batches % 500 == 0:
        sanity_file_paths.append((count_batches, fpaths[0]))
        # GT bounding boxes
        vis_gt_pred(images[0], boxes[0], labels[0], pred_cpu[0], tmp_path=snv3_tmp, batch_num=count_batches)

pprint(sanity_file_paths)

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

