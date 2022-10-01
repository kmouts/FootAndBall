#
# Train FootAndBall detector on ISSIA-CNR Soccer and SoccerPlayerDetection dataset
#


import argparse

import numpy as np
import os
import time

import torch


from data.spd_bmvc2017_dataset import create_spd_dataset
from misc.config import Params

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

'''
NOTES:
In camera 5 some of the player bboxes are moved by a few pixels from the true position.
When evaluating mean precision use smaller IoU ratio, otherwise detection results are poor.
Alternatively add some margin to ISSIA ground truth bboxes.
'''
