import pickle
import sys

import cv2
import numpy as np

from data.issia_utils import _annotate_frame
from data.issia_dataset import create_issia_dataset
from data.augmentation import tensor2image, PLAYER_LABEL

if __name__ == '__main__':
    print('Check issia Dataset')
    sys.path.append('data')
    print(sys.path)

    dataset_path = '/mnt/DATA/DATASETS/ISSIA-CNR/issia/'
    camera_id = 1
    image_id = 1272

    folder_dir = dataset_path + 'unpacked/' + str(camera_id) + '/'
    anns = pickle.load(open(dataset_path + "issia_gt_anns.p", "rb"))
    frame = cv2.imread(folder_dir + str(image_id) + '.png')
    frame = _annotate_frame(frame, image_id, anns[camera_id - 1], color=(0, 0, 255))
    cv2.imshow('frame', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
    cv2.imwrite(dataset_path + 'test/' + str(image_id) + '_from_file.png', frame)
    key = cv2.waitKey(1) & 0xFF
    cv2.waitKey()

    # try now the issia_dataset module
    mode = 'val'
    dataset = create_issia_dataset(dataset_path, [camera_id], mode, only_ball_frames=False)
    anns2 = dataset.gt_annotations[camera_id].persons[image_id]
    assert (anns[0].persons[image_id] == anns2)  # make sure pickle anns are the same
    dboxes, dlabels = dataset.get_annotations(1, image_id)  # players and ball(s)
    # find the corresponding index in the image_list
    image_ndx = dataset.image_list.index((folder_dir + str(image_id) + '.png', camera_id, image_id))
    image, boxes, labels = dataset[image_ndx]
    denormalised_image = 255 * tensor2image(image)
    # denormalised_image = np.asarray(denormalised_image, dtype="uint")
    int_boxes = boxes.numpy().astype(int).tolist()
    for b in int_boxes:
        cv2.rectangle(denormalised_image, (b[0], b[1]), (b[2], b[3]), color=(0, 0, 255))
    denormalised_image = np.asarray(denormalised_image, dtype="uint")
    # cv2.imshow('denormalised_frame', cv2.resize(denormalised_image, (0, 0), fx=0.5, fy=0.5))
    cv2.imwrite(dataset_path + 'test/' + str(image_id) + '_denormalised.png', denormalised_image)
    key = cv2.waitKey(1) & 0xFF
    cv2.waitKey()
    cv2.destroyAllWindows()

    player_boxes = [b for ind, b in enumerate(int_boxes) if labels.tolist()[ind] == PLAYER_LABEL]

