import os
# os.system('nvidia-smi')
import sys
import cv2
import torch

sys.path.append('/opt/project')

from network import footandball
from data.issia_utils import read_issia_ground_truth

print(torch.cuda.get_device_name(0))
torch.cuda.init()
model_name = 'fb1'
model_weights_path = 'models/model_20201019_1416_final.pth'
ball_confidence_threshold = 0.7
player_confidence_threshold = 0.7
my_device = 'cuda'
issia_video_camera_number = 5
video_path = '/DATASETS/ISSIA-CNR/issia/filmrole' + str(issia_video_camera_number) + '.avi'
issia_dataset_path = '/DATASETS/ISSIA-CNR/issia/'

print(video_path)
assert os.path.isfile(video_path)
print(issia_dataset_path)
assert os.path.isdir(issia_dataset_path)

model = footandball.model_factory(model_name, 'detect', ball_threshold=ball_confidence_threshold,
                                  player_threshold=player_confidence_threshold)

model.print_summary(show_architecture=False)
model = model.to(my_device)

_, file_name = os.path.split(video_path)

if my_device == 'cpu':
    print('Loading CPU weights...')
    state_dict = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
else:
    print('Loading GPU weights...')
    state_dict = torch.load(model_weights_path)

model.load_state_dict(state_dict)
# Set model to evaluation mode
model.eval()

sequence = cv2.VideoCapture(video_path)

sequence.release()

gt_annotations = read_issia_ground_truth(5, issia_dataset_path)

# gt_annotations.persons=defaultdict(<class 'list'>, {357: [('0', 81, 21, 1897, 208),
# ('8', 75, 34, 1675, 12), ('7', 97, 48, 713, 275)],
print('End for evaluate.py')
