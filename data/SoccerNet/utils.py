from collections import OrderedDict
from pathlib import Path
import json
import os

import cv2
import numpy as np

from data.augmentation import tensor2image


def getListGames(split="v1", task="spotting"):
    #  if single split, convert into a list
    if not isinstance(split, list):
        split = [split]

    # if an element is "v1", convert to  train/valid/test
    if "all" in split:
        split = ["train", "valid", "test", "challenge"]
        if task == "frames":
            split = ["train", "valid", "test"]
    if "v1" in split:
        split.pop(split.index("v1"))
        split.append("train")
        split.append("valid")
        split.append("test")

    # if task == "spotting":

    listgames = []
    # print(split)
    # loop over splits
    for spl in split:
        if task == "spotting":
            if spl == "train":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetGamesTrain.json"
            elif spl == "valid":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetGamesValid.json"
            elif spl == "test":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetGamesTest.json"
            elif spl == "challenge":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetGamesChallenge.json"

        elif task == "camera-changes":
            if spl == "train":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetCameraChangesTrain.json"
            elif spl == "valid":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetCameraChangesValid.json"
            elif spl == "test":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetCameraChangesTest.json"
            elif spl == "challenge":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetCameraChangesChallenge.json"


        elif task == "frames":
            if spl == "train" or spl == "rgb_train":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetFramesTrain.json"
            elif spl == "valid":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetFramesValid.json"
            elif spl == "test":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetFramesTest.json"
            elif spl == "challenge":
                jsonGamesFile = Path(__file__).parent / "data/SoccerNetFramesChallenge.json"

        with open(jsonGamesFile, "r") as json_file:
            dictionary = json.load(json_file, object_pairs_hook=OrderedDict)  # ensure same filelist order

        for championship in dictionary:
            for season in dictionary[championship]:
                for game in dictionary[championship][season]:
                    listgames.append(os.path.join(championship, season, game))

    return listgames


def vis_gt_pred(image, box, label, pred, tmp_path, batch_num,  snv3=False):

    # frame = tensor2image(image,  snv3=snv3) * 255

    frame = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    for (x1, y1, x2, y2), lb in zip((np.rint(box.numpy(force=True))).astype(int), label.numpy(force=True)):
        colors = [(0, 0, 153), (204, 0, 0)]
        if lb == 1:  # ball
            color = colors[0]
        elif lb == 2:  # player
            color = colors[1]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

    for (x1, y1, x2, y2), lb in zip((np.rint(pred['boxes'].numpy(force=True))).astype(int),
                                    pred['labels'].numpy(force=True)):
        colors = [(51, 255, 255), (255, 153, 51)]
        if lb == 1:  # ball
            color = colors[0]
        elif lb == 2:  # player
            color = colors[1]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        cv2.imshow('frame', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))

    cv2.imwrite(tmp_path + 'batch_' + str(batch_num) + '_0.png', frame)


if __name__ == "__main__":
    print(len(getListGames(["v1"], task="camera-changes")))
