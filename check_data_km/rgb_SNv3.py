import os
# os.system('nvidia-smi')
import sys

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.SNv3_dataloader import create_snv3_dataset
from data.data_reader import my_collate
import matplotlib.pyplot as plt

sys.path.append('..')
print(torch.cuda.get_device_name(0))
torch.cuda.init()

only_ball_frames = False
my_device = 'cuda'
snv3_dataset_path = '/mnt/DATA/DATASETS/SOCCERNETv3/SNV3/SNV3_PIP_data_final'
snv3_tmp = '/mnt/DATA/DATASETS/SOCCERNETv3/tmp/'

rgb_train_snv3_dataset = create_snv3_dataset(snv3_dataset_path, tmode='rgb_train', only_ball_frames=only_ball_frames,
                                             preload_images=True)
dataloaders = {'rgb_train': DataLoader(rgb_train_snv3_dataset, batch_size=4, num_workers=2,
                                       pin_memory=True, collate_fn=my_collate)}

print('\n Images: {}'.format(len(rgb_train_snv3_dataset)))
print('\nTrain set: Dataset size (in batches): {}'.format(len(dataloaders['rgb_train'])))
phase = 'rgb_train'
count_batches = 0

# https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
# # placeholders
psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# Iterate over data.
for ndx, (images, boxes, labels, fpaths) in enumerate(tqdm(dataloaders[phase])):
    count_batches += 1

    if ndx == 0:
        fig = plt.figure(figsize=(14, 7))
        for i in range(4):
            ax = fig.add_subplot(1, 4, i + 1, xticks=[], yticks=[])
            plt.imshow(images[i].numpy().transpose(1, 2, 0))
        plt.show()

    psum += images.sum(axis=[0, 2, 3])
    psum_sq += (images ** 2).sum(axis=[0, 2, 3])

# FINAL CALCULATIONS
# pixel count
train_image_size = (720, 1280)
count = len(rgb_train_snv3_dataset) * train_image_size[0] * train_image_size[1]

# mean and std
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean ** 2)
total_std = torch.sqrt(total_var)

# output
print('mean: ' + str(total_mean))
print('std:  ' + str(total_std))


# https://www.binarystudy.com/2021/04/how-to-calculate-mean-standard-deviation-images-pytorch.html
def batch_mean_and_sd(loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _, _, _ in tqdm(loader):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return mean, std


mean, std = batch_mean_and_sd(dataloaders[phase])
print("mean and std: \n", mean, std)
