import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import zipfile
from PIL import Image
from tqdm import tqdm
import os
import json

from data import augmentation
from data.augmentation import BALL_LABEL, PLAYER_LABEL
from .SoccerNet.Evaluation.utils import FRAME_CLASS_DICTIONARY
import copy
import time
from .SoccerNet.utils import getListGames
from argparse import ArgumentParser


def create_snv3_dataset(dataset_path, tmode, only_ball_frames=False, tiny=None, preload_images=False, meta_only=False,
                        max_player_height=0):
    # Get SoccerNet v3 Dataset
    assert tmode == 'train' or tmode == 'valid' or tmode == 'test' or tmode == 'rgb_train'
    assert os.path.exists(dataset_path), 'Cannot find dataset: ' + str(dataset_path)

    train_image_size = (720, 1280)
    val_image_size = (720, 1280)
    test_image_size = (720, 1280)

    if tmode == 'train':
        transform = augmentation.TrainAugmentation(size=train_image_size, SNv3=True)
    elif tmode == 'valid':
        transform = augmentation.NoAugmentation(size=val_image_size, SNv3=True)
    elif tmode == 'test':
        transform = augmentation.NoAugmentation(size=test_image_size, SNv3=True)
    elif tmode == 'rgb_train':
        transform = augmentation.NoAugmentationForRGB(size=test_image_size)

    dataset = SNV3Dataset(dataset_path, transform, preload_images, split=tmode, only_ball_frames=only_ball_frames,
                          tiny=tiny, meta_only=meta_only, max_player_height=max_player_height)
    return dataset


class SNV3Dataset(Dataset):

    def __init__(self, path, transform, only_ball=False, split="all", resolution=(720, 1280), preload_images=False,
                 tiny=None, meta_only=False,
                 zipped_images=True, only_ball_frames=False, max_player_height=0):

        self.transform = transform

        self.only_ball_frames = only_ball_frames
        self.meta_only = meta_only

        # Path for the SoccerNet-v3 dataset
        # containing the images and labels
        self.path = path

        # Get the list of the selected subset of games
        self.list_games = getListGames(split, task="frames")
        if tiny is not None:
            self.list_games = self.list_games[:tiny]

        self.preload_images = preload_images
        self.zipped_images = zipped_images

        # Variable to store the metadata
        print("Reading the annotation files")
        self.metadata = list()
        for game in tqdm(self.list_games):
            self.metadata.append(json.load(open(os.path.join(self.path, game, "Labels-v3.json"))))

        # Variables to store the preloaded images and annotations
        # Each element in the list WAS a list of images and annotations linked to an action
        # now each element is one data_tmp dict
        self.data = list()
        for annotations in tqdm(self.metadata):

            # Retrieve each action in the game
            for action_name in annotations["GameMetadata"]["list_actions"]:

                # concatenate the replays of each action with itself
                img_list = [action_name] + annotations["actions"][action_name]["linked_replays"]

                IDs_list = list()

                zipfilepath = os.path.join(self.path, annotations["GameMetadata"]["UrlLocal"], 'Frames-v3.zip')
                zippedFrames = None
                if self.zipped_images:
                    zippedFrames = zipfile.ZipFile(zipfilepath, 'r')

                # For each IMAGE extract the IMAGES(! replays..) and annotations
                for i, img in enumerate(img_list):

                    # Variable to save the annotation
                    data_tmp = dict()
                    data_tmp["image"] = None

                    # Only the first frame is an action, the rest are replays
                    img_type = "actions"
                    if i > 0:
                        img_type = "replays"

                    filepath = os.path.join(self.path, annotations["GameMetadata"]["UrlLocal"], "v3_frames", img)
                    if self.preload_images:
                        with torch.no_grad():
                            if self.zipped_images:
                                imginfo = zippedFrames.open(img)
                                data_tmp["image"] = transforms.ToTensor()(Image.open(imginfo)) * 255
                            else:
                                data_tmp["image"] = torchvision.io.read_image(filepath)

                    data_tmp["zipfilepath"] = zipfilepath
                    data_tmp["imagefilepath"] = img
                    data_tmp["filepath"] = filepath

                    data_tmp["bboxes"], data_tmp["labels"], \
                        data_tmp["ball_in_frame"], data_tmp["max_height"], \
                        data_tmp["im_dim"] = self.format_bboxes(annotations[img_type][img]["bboxes"],
                                                                annotations[img_type][img]["imageMetadata"])

                    # filter out image with larger player height
                    if max_player_height and data_tmp["max_height"] > max_player_height:
                        continue

                    if self.only_ball_frames and not data_tmp["ball_in_frame"]:
                        continue

                    self.data.append(data_tmp)

    def format_bboxes(self, bboxes, image_metadata):

        # Bounding boxes in x_top, y_top, x_right_down, y_right_down
        boxes = list()

        labels = list()
        ball_in_frame = False
        max_height = 0
        im_dim = (int(image_metadata["height"]), int(image_metadata["width"]))

        for i, bbox in enumerate(bboxes):
            bboxc = bbox["class"]
            if bboxc is not None:
                if bboxc == "Ball":
                    ball_in_frame = True
                    labels.append(BALL_LABEL)
                    boxes.append(
                        (bbox["points"]["x1"], bbox["points"]["y1"], bbox["points"]["x2"], bbox["points"]["y2"]))
                elif "Player" in bboxc or "Goalkeeper" in bboxc or "referee" in bboxc or "Staff" in bboxc:
                    labels.append(PLAYER_LABEL)
                    boxes.append(
                        (bbox["points"]["x1"], bbox["points"]["y1"], bbox["points"]["x2"], bbox["points"]["y2"]))
                    player_height = bbox["points"]["y2"] - bbox["points"]["y1"]
                    if player_height > max_height:
                        max_height = player_height

        return boxes, labels, ball_in_frame, max_height, im_dim

    def __getitem__(self, index):

        if not self.preload_images:
            local_data = copy.deepcopy(self.data[index])
            with torch.no_grad():
                image_list = list()

                image = []
                if not self.meta_only:
                    if self.zipped_images:
                        imginfo = zipfile.ZipFile(local_data["zipfilepath"], 'r').open(local_data["imagefilepath"])
                        image = Image.open(imginfo)
                    else:
                        image = Image.open(local_data["filepath"])

                boxes, labels = np.array(local_data["bboxes"], dtype=np.float), \
                    np.array(local_data["labels"], dtype=np.int64)
                if not self.meta_only:
                    image, boxes, labels = self.transform((image, boxes, labels))
                boxes = torch.tensor(boxes, dtype=torch.float)
                labels = torch.tensor(labels, dtype=torch.int64)

                local_data["image"] = image
                local_data["bboxes"] = boxes
                local_data["labels"] = labels

                return local_data["image"], local_data["bboxes"], local_data["labels"], \
                    local_data["filepath"], local_data["im_dim"], local_data["max_height"], \
                    local_data["ball_in_frame"]

        return self.data[index]

    def __len__(self):

        return len(self.data)


if __name__ == "__main__":

    # Load the arguments
    parser = ArgumentParser(description='dataloader')

    parser.add_argument('--SoccerNet_path', required=True, type=str, help='Path to the SoccerNet-V3 dataset folder')
    parser.add_argument('--tiny', required=False, type=int, default=None, help='Select a subset of x games')
    parser.add_argument('--split', required=False, type=str, default="all", help='Select the split of data')
    parser.add_argument('--num_workers', required=False, type=int, default=4,
                        help='number of workers for the dataloader')
    parser.add_argument('--resolution_width', required=False, type=int, default=1280,
                        help='width resolution of the images')
    parser.add_argument('--resolution_height', required=False, type=int, default=720,
                        help='height resolution of the images')
    parser.add_argument('--preload_images', action='store_true',
                        help="Preload the images when constructing the dataset")
    parser.add_argument('--zipped_images', action='store_true', help="Read images from zipped folder")

    args = parser.parse_args()

    start_time = time.time()
    soccernet = SNV3Dataset(args.SoccerNet_path, split=args.split,
                            resolution=(args.resolution_width, args.resolution_height),
                            preload_images=args.preload_images, zipped_images=args.zipped_images, tiny=args.tiny)
    soccernet_loader = DataLoader(soccernet, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    with tqdm(enumerate(soccernet_loader), total=len(soccernet_loader), ncols=160) as t:
        for i, data in t:
            continue
    end_time = time.time()
    print(end_time - start_time)
