from pathlib import Path
import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from tqdm import tqdm
import matplotlib

from data.SNv3_dataloader import create_snv3_dataset
from data.augmentation import tensor2image
from utils import getListGames

matplotlib.rcParams['savefig.dpi'] = 1200
matplotlib.rcParams["figure.dpi"] = 100

plt.style.use('ggplot')
plt.figure(dpi=1200)
fig, axes = plt.subplots(nrows=3, ncols=3, layout="constrained")
# fig = plt.gcf()


# snv3_dataset_path = '/mnt/DATA/DATASETS/SOCCERNETv3/SNV3/SNV3_PIP_data_final'
snv3_dataset_path = '/home/kmouts/Projects/SNV3/SNV3_PIP_data_final'
only_ball_frames = False

modes = ["train", "valid", "test"]
colors = ["green", "orange", "red"]

for i, mode in enumerate(modes):
    list_games = getListGames(split=mode, task="frames")
    snv3_dataset = create_snv3_dataset(snv3_dataset_path, tmode=mode, only_ball_frames=only_ball_frames,
                                       meta_only=True)

    images_dim = [None] * len(snv3_dataset)
    players_max_height = [None] * len(snv3_dataset)
    ball_in_frame = [None] * len(snv3_dataset)

    for ind, data_im in enumerate(tqdm(snv3_dataset)):
        images_dim[ind] = str(data_im[4])
        players_max_height[ind] = data_im[5]
        ball_in_frame[ind] = data_im[6]

    images_dim.sort()

    axes[i, 0].hist(images_dim, color=colors[i])
    axes[i, 0].set_ylabel('# of images', fontsize=8)
    # axes[i, 0].yaxis.set_major_locator(ticker.MultipleLocator(5000))
    axes[i, 0].tick_params(axis='y', labelsize=5)
    axes[i, 0].tick_params(axis='x', labelsize=6, rotation=90)
    if i == 0:
        axes[i, 0].set_title("Image dimensions", fontsize=10)

    axes[i, 1].hist(players_max_height, color=colors[i])
    axes[i, 1].tick_params(axis='y', labelsize=5)
    axes[i, 1].tick_params(axis='x', labelsize=6)
    axes[i, 1].set_ylabel('# of players', fontsize=8)
    axes[i, 1].set_xlabel('Height in pixels', fontsize=8)
    if i == 0:
        axes[i, 1].set_title("Max Player Height", fontsize=10)

    axes[i, 2].hist(np.array(ball_in_frame, dtype=int), bins=[0, 0.5, 1], rwidth=0.8, color=colors[i])
    axes[i, 2].set_xticks(ticks=[0.3, 0.8], labels=['no ball', 'with ball'])
    axes[i, 2].tick_params(axis='y', labelsize=5)
    axes[i, 2].tick_params(axis='x', labelsize=8)
    axes[i, 2].set_ylabel('# of images', fontsize=8)
    if i == 0:
        axes[i, 2].set_title('Frames with ball', fontsize=10)

fig.suptitle('SoccerNet v3 Stats', fontsize=16)
# fig.tight_layout()
plt.savefig('SoccerNetv3_stats.pdf', dpi=1200)
plt.show()

# if __name__ == "__main__":
#     print(len(getListGames(["v1"], task="camera-changes")))
