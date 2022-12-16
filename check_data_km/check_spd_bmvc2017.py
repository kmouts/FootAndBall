#
# Train FootAndBall detector on ISSIA-CNR Soccer and SoccerPlayerDetection dataset
#


import argparse

import cv2
import numpy as np
import os
import time

# import torch

import sys

sys.path.append('/opt/project/')
sys.path.append('/opt/project/data')

from data.spd_bmvc2017_dataset import create_spd_dataset
from misc.config import Params

print(sys.path)

if __name__ == '__main__':

    print('Check spd_bmvc2017 Dataset')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to the configuration file', type=str, default='../config.txt')
    parser.add_argument('--debug', dest='debug', help='debug mode', action='store_true')
    args = parser.parse_args()

    print('Config path: {}'.format(args.config))
    print('Debug mode: {}'.format(args.debug))

    params = Params(args.config)
    params.print()

    train_spd_dataset = create_spd_dataset(params.spd_path, params.spd_set, mode='train')

    # Read First Image
    for idx, img in enumerate(train_spd_dataset.datasets[0].image_list):
        img1 = cv2.imread(img)

        # Person bounding boxes
        for (x1, y1, x2, y2) in train_spd_dataset.datasets[0].gt[idx]:
            cv2.rectangle(img1, (x1.astype(int), y1.astype(int)), (x2.astype(int), y2.astype(int)), color=(0, 0, 255))
        cv2.imshow('annotated image', img1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey()

    cv2.destroyAllWindows()

'''
NOTES:
In camera 5 some of the player bboxes are moved by a few pixels from the true position.
When evaluating mean precision use smaller IoU ratio, otherwise detection results are poor.
Alternatively add some margin to ISSIA ground truth bboxes.
'''
