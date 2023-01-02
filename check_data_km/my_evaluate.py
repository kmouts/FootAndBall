import os
import pickle
# os.system('nvidia-smi')
import sys
import cv2
import torch
import tqdm as tqdm
from torch.utils.data import DataLoader

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


cameras = [5]
only_ball_frames = False

# https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training-ed127d825db7
val_issia_dataset = create_issia_dataset(issia_dataset_path, cameras, mode='val', only_ball_frames=only_ball_frames)
dataloaders = {'val': DataLoader(val_issia_dataset, batch_size=2, num_workers=8,
                                 pin_memory=True, collate_fn=my_collate)}

print('Validation set: Dataset size: {}'.format(len(dataloaders['val'].dataset)))

# model = footandball.model_factory(model_name, 'detect', ball_threshold=ball_confidence_threshold,
#                                   player_threshold=player_confidence_threshold)
# model.print_summary(show_architecture=False)
# model = model.to(my_device)
#
# if my_device == 'cpu':
#     print('Loading CPU weights...')
#     state_dict = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
# else:
#     print('Loading GPU weights...')
#     state_dict = torch.load(model_weights_path)
#
# model.load_state_dict(state_dict)
# Set model to evaluation mode

phase = 'val'
# model.eval()

# code from "train_detector"

batch_stats = {'loss': [], 'loss_ball_c': [], 'loss_player_c': [], 'loss_player_l': []}

# count_batches = 0
# # Iterate over data.
# for ndx, (images, boxes, labels) in enumerate(dataloaders[phase]):
#     images = images.to(my_device)
#     h, w = images.shape[-2], images.shape[-1]
#     gt_maps = model.groundtruth_maps(boxes, labels, (h, w))
#     gt_maps = [e.to(my_device) for e in gt_maps]
#     count_batches += 1
#
#     predictions = model(images)



from time import time
import multiprocessing as mp

for num_workers in range(2, mp.cpu_count(), 2):
    train_loader = DataLoader(val_issia_dataset,shuffle=True,num_workers=num_workers,
                              batch_size=20,pin_memory=True,collate_fn=my_collate)
    start = time()
    for epoch in range(1, 2): # (1,3)
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))




