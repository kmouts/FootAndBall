import os
# os.system('nvidia-smi')
import sys
import cv2
import torch

sys.path.append('/opt/project')
print(torch.cuda.get_device_name(0))

from network import footandball

model_name = 'fb1'
model_weights_path = 'models/model_20201019_1416_final.pth'
ball_confidence_threshold = 0.7
player_confidence_threshold = 0.7
my_device = 'cuda'
video_path = '/DATASETS/ISSIA-CNR/issia/filmrole5.avi'

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

print('End for evaluate.py')
