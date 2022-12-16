#
# Train FootAndBall detector on ISSIA-CNR Soccer and SoccerPlayerDetection dataset
#


import argparse

import cv2
import numpy as np
import os
import time
import sys

# import torch
from data.issia_utils import visualize_issia_gt_images


from data.spd_bmvc2017_dataset import create_spd_dataset
from misc.config import Params

if __name__ == '__main__':
    print('Check issia Dataset')
    sys.path.append('/opt/project/data')
    print(sys.path)

    visualize_issia_gt_images(1, '/DATASETS/ISSIA-CNR/issia/')
'''
NOTES:
In camera 5 some of the player bboxes are moved by a few pixels from the true position.
When evaluating mean precision use smaller IoU ratio, otherwise detection results are poor.
Alternatively add some margin to ISSIA ground truth bboxes.
'''
