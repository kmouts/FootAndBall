from pathlib import Path
import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

from data.SNv3_dataloader import create_snv3_dataset
from data.augmentation import tensor2image
from utils import getListGames

# snv3_dataset_path = '/mnt/DATA/DATASETS/SOCCERNETv3/SNV3/SNV3_PIP_data_final'
snv3_dataset_path = '/home/kmouts/Projects/SNV3/SNV3_PIP_data_final'
only_ball_frames = False

modes = ["train", "valid", "test"]



list_games = getListGames(split=modes[0], task="frames")
snv3_dataset = create_snv3_dataset(snv3_dataset_path, tmode=modes[0], only_ball_frames=only_ball_frames,
                                   meta_only=True)

images_dim = [None] * len(snv3_dataset)
players_max_height = [None] * len(snv3_dataset)
ball_in_frame = [None] * len(snv3_dataset)

for ind, data_im in enumerate(tqdm(snv3_dataset)):
    images_dim[ind] = str(data_im[4])
    players_max_height[ind] = data_im[5]
    ball_in_frame[ind] = data_im[6]

# Test with creating a new dataset
list_games2 = getListGames(split=modes[0], task="frames")
assert list_games2 == list_games
snv3_dataset = create_snv3_dataset(snv3_dataset_path, tmode=modes[0], only_ball_frames=only_ball_frames,
                                   meta_only=True)

images_dim2 = [None] * len(snv3_dataset)
players_max_height2 = [None] * len(snv3_dataset)
ball_in_frame2 = [None] * len(snv3_dataset)

for ind, data_im in enumerate(tqdm(snv3_dataset)):
    images_dim2[ind] = str(data_im[4])
    players_max_height2[ind] = data_im[5]
    ball_in_frame2[ind] = data_im[6]

assert images_dim2 == images_dim
assert players_max_height2 == players_max_height
assert ball_in_frame2 == ball_in_frame

# Test with the same dataset (as in epochs)
images_dim3 = [None] * len(snv3_dataset)
players_max_height3 = [None] * len(snv3_dataset)
ball_in_frame3 = [None] * len(snv3_dataset)

for ind, data_im in enumerate(tqdm(snv3_dataset)):
    images_dim3[ind] = str(data_im[4])
    players_max_height3[ind] = data_im[5]
    ball_in_frame3[ind] = data_im[6]

assert images_dim2 == images_dim3
assert players_max_height2 == players_max_height3
assert ball_in_frame2 == ball_in_frame3


# if __name__ == "__main__":
#     print(len(getListGames(["v1"], task="camera-changes")))
