

# import torch
# # import pandas as pd


# # class SoccerNetDataLoaderTorch(torch.utils.data.Dataset):
    
# #     def __init__(self, soccernet_dir, jsonGamesFile=None, transform=None, input="Video_LQ"):
# #         """
# #         Args:
# #             csv_file (string): Path to the csv file with annotations.
# #             root_dir (string): Directory with all the images.
# #             transform (callable, optional): Optional transform to be applied
# #                 on a sample.
# #         """

# #         if jsonGamesFile is None:
# #             jsonGamesFile = Path(__file__).parent / "data/SoccerNetGames.json"

# #         self.jsonGamesFile = jsonGamesFile

# #         with open(jsonGamesFile, "r") as json_file:
# #             dictionary = json.load(json_file)

# #         self.games = []
# #         for championship in dictionary:
# #             for season in dictionary[championship]:
# #                 for game in dictionary[championship][season]:

# #                     self.games.append(os.path.join(championship, season, game))
                    

# #         # self.landmarks_frame = pd.read_csv(csv_file)
# #         self.soccernet_dir = soccernet_dir
# #         self.transform = transform
# #         self.input = input

# #     def __len__(self):
# #         return len(self.games)

# #     def __getitem__(self, idx):
# #         game = self.games[idx]

# #         if self.input == "Video_LQ":
# #             labels = scikit.io.vread(os.path.join(self.games[idx], "Labels.txt")
# #             video1 = scikit.io.vread(os.path.join(self.games[idx], "1.mkv")
# #             video2 = scikit.io.vread(os.path.join(self.games[idx], "2.mkv")

# #             sample={'video1': video1, 'video2': video2,
# #                     'labels': labels}

# #             if self.transform:
# #                 sample = self.transform(sample)

# #         return sample


# import torch
# import torchvision
# import math
# import os
# import argparse
# import numpy as np
# import cv2  # pip install opencv-python (==3.4.11.41)
# import imutils  # pip install imutils
# # import skvideo.io
# from tqdm import tqdm
# import skvideo.io


# def getDuration(video_path):
#     metadata = skvideo.io.ffprobe(video_path)

#     # read "@duration" metadata if availabel
#     try:
#         time_second = float(metadata["video"]["@duration"])
#     except:
#         pass

#     # prioritize "video" "tag" "DURATION" metadata if readily available
#     for entry in metadata["video"]["tag"]:
#         if list(entry.items())[0][1] == "DURATION":
#             # print("entry", entry)
#             duration = list(entry.items())[1][1].split(":")
#             # print(duration)
#             time_second = int(duration[0])*3600 + \
#                 int(duration[1])*60 + float(duration[2])

#     # print("duration video", time_second, video_path)
#     return time_second


# def video2clips(video, stride, clip_len):
#     # nFrames = video.shape[0]
#     start_idx = 0  # math.ceil(-clip_len/2)
#     end_idx = video.shape[0] # - math.ceil(clip_len/2)
#     idx = torch.arange(start=start_idx, end=end_idx, step=stride).long()
#     idxs = []
#     for i in torch.arange(math.ceil(-clip_len/2), math.ceil(clip_len/2)):
#         idxs.append(idx+i)
#     idx = torch.stack(idxs, dim=1)
#     # idx = unfold2(idx, size=clip_len, step=stride)
#     # print(idx)
#     idx = idx.clamp(0, video.shape[0]-1)
#     # print(idx)
#     return video[idx]
#     # i_frames = [for i in range()]
#     # for i in range(nFrames):
#     #     if i

#     #     print(frame[])

# # def resize(vid, size, interpolation='bilinear'):
# #     # NOTE: using bilinear interpolation because we don't work on minibatches
# #     # at this level
# #     scale = None
# #     if isinstance(size, int):
# #         scale = float(size) / min(vid.shape[-2:])
# #         size = None
# #     return torch.nn.functional.interpolate(
# #         vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)


# def crop(vid, i, j, h, w):
#     return vid[..., i:(i + h), j:(j + w)]


# def center_crop(vid, output_size):
#     h, w = vid.shape[-2:]
#     th, tw = output_size

#     i = int(round((h - th) / 2.))
#     j = int(round((w - tw) / 2.))
#     return crop(vid, i, j, th, tw)


# def hflip(vid):
#     return vid.flip(dims=(-1,))


# # NOTE: for those functions, which generally expect mini-batches, we keep them
# # as non-minibatch so that they are applied as if they were 4d (thus image).
# # this way, we only apply the transformation in the spatial domain
# def resize(vid, size, interpolation='bilinear'):
#     # NOTE: using bilinear interpolation because we don't work on minibatches
#     # at this level
#     scale = None
#     if isinstance(size, int):
#         scale = float(size) / min(vid.shape[-2:])
#         size = None
#     return torch.nn.functional.interpolate(
#         vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)


# def pad(vid, padding, fill=0, padding_mode="constant"):
#     # NOTE: don't want to pad on temporal dimension, so let as non-batch
#     # (4d) before padding. This works as expected
#     return torch.nn.functional.pad(vid, padding, value=fill, mode=padding_mode)


# def to_normalized_float_tensor(vid):
#     return vid.permute(3, 0, 1, 2).to(torch.float32) / 255


# def normalize(vid, mean, std):
#     shape = (-1,) + (1,) * (vid.dim() - 1)
#     # shape = (1,) * (vid.dim() - 1) + (-1,)
#     # print(vid.shape)
#     mean = torch.as_tensor(mean).reshape(shape)
#     # print(mean.shape)
#     std = torch.as_tensor(std).reshape(shape)
#     # print(std.shape)
#     return (vid - mean) / std


# class FrameDataset(torch.utils.data.Dataset):
#     def __init__(self, video_path, FPS_desired):
#         self.video_path=video_path
#         self.FPS_desired = FPS_desired
#         # self.clip_len = clip_len        

#         # vframes, aframes, info = torchvision.io.read_video(
#         #     video_path, pts_unit='sec')
#         videodata = skvideo.io.FFmpegReader(video_path)
#         (numframe, _, _, _) = videodata.getShape()
#         videodata = skvideo.io.vreader(video_path)
#         frames = []
#         self.duration = getDuration(video_path)
#         self.FPS = numframe/self.duration
#         stride = self.FPS/self.FPS_desired
#         for i_frame, frame in tqdm(enumerate(videodata), total=numframe):
#             # frame = imutils.resize(frame, height=112)
#             if (i_frame % stride < 1):
#                 # print(frame.shape)
#                 frame = imutils.resize(frame, height=224)  # keep aspect ratio
#                 # number of pixel to remove per side
#                 off_side = int((frame.shape[1] - 224)/2)
#                 frame = frame[:, off_side:-off_side, :]
#                 # print(frame.shape)
#                 # clip = resize(clip, (112, 112))
#                 # print("max frame intensity", np.max(frame))
#                 frames.append(frame)
#         frames = np.array(frames)
#         vframes = torch.from_numpy(frames)
#         print(vframes.shape)
#         vframes = vframes.permute([0, 3, 1, 2])


#         print(vframes.shape)
#         # normalize video
#         # print("max vframes intensity", vframes.max())
#         self.clips = vframes/255.0
#         # print("max self.clips intensity", self.clips.max())
#         # self.clips = normalize(vid=self.clips,
#         #                     mean=[0.485, 0.456, 0.406],
#         #                     std=[0.229, 0.224, 0.225])
#         # shape = (1,)*(vframes.dim()-1) + (-1,)
#         # mean = torch.as_tensor([0.43216, 0.394666, 0.37645]).reshape(shape)
#         # std = torch.as_tensor([0.22803, 0.22145, 0.216989]).reshape(shape)
#         # vframes = (vframes/255.0-mean)/std

#         # create clips
#         # self.clips = video2clips(vframes, stride=stride, clip_len=1)
        
#         # permute and resize at need
#         # print(vframes.shape)
#         # self.clips = vframes.permute([0, 3, 1, 2]) #[:,:,0,:,:]
#         # print(self.clips.shape)

#     def __len__(self):
#         return len(self.clips)



#     def __getitem__(self, idx):
#         # transform = torch.nn.functional.interpolate(
#             # vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)
#         clip = self.clips[idx, :, :, :]
#         # print("clip", clip.shape)
#         # clip = clip[:, :, ::2, 87:-87:2]
#         # print(clip.shape)
#         # clip = resize(clip, (224,224))
#         # print(clip.shape)
#         return clip


# class ClipDataset(torch.utils.data.Dataset):
#     def __init__(self, video_path, FPS_desired, clip_len):
#         self.video_path=video_path
#         self.FPS_desired = FPS_desired
#         self.clip_len = clip_len        

#         # vframes, aframes, info = torchvision.io.read_video(
#         #     video_path, pts_unit='sec')
#         videodata = skvideo.io.FFmpegReader(video_path)
#         (numframe, _, _, _) = videodata.getShape()
#         videodata = skvideo.io.vreader(video_path)
#         frames = []
#         for i_frame, frame in tqdm(enumerate(videodata), total=numframe):
#             frame = imutils.resize(frame, height=224)
#             off_side = int((frame.shape[1] - 224)/2)
#             frame = frame[:, off_side:off_side+224, :]
#             frame = imutils.resize(frame, height=112)  # keep aspect ratio
#             # number of pixel to remove per side
#             # clip = resize(clip, (112, 112))
#             # print("max frame intensity", np.max(frame))
#             frames.append(frame)
#         frames = np.array(frames)
#         vframes = torch.from_numpy(frames)

#         self.duration = getDuration(video_path)
#         self.FPS = vframes.shape[0]/self.duration

#         # normalize video
#         vframes = vframes/255.0
#         # vframes = normalize(vid=vframes,
#         #                     mean=[0.43216, 0.394666, 0.37645],
#         #                     std=[0.22803, 0.22145, 0.216989])
#         # shape = (1,)*(vframes.dim()-1) + (-1,)
#         # mean = torch.as_tensor([0.43216, 0.394666, 0.37645]).reshape(shape)
#         # std = torch.as_tensor([0.22803, 0.22145, 0.216989]).reshape(shape)
#         # vframes = (vframes/255.0-mean)/std

#         # create clips
#         stride = self.FPS/self.FPS_desired
#         self.clips = video2clips(
#             vframes, stride=stride, clip_len=self.clip_len)
#         # print(self.clips.shape)
        
#         # permute and resize at need
#         self.clips = self.clips.permute([0, 4, 1, 2, 3])
#         # print(self.clips.shape)

#     def __len__(self):
#         return len(self.clips)



#     def __getitem__(self, idx):
#         # transform = torch.nn.functional.interpolate(
#             # vid, size=size, scale_factor=scale, mode=interpolation, align_corners=False)
#         clip = self.clips[idx, ...]
#         # clip = clip[:, :, ::2, 87:-87:2]
#         # print(clip.shape)
#         # clip = resize(clip, (112,112))
#         # print(clip.shape)
#         return clip


# if __name__ == "__main__":
#     dataset = ClipDataset(
#         "/media/giancos/Football/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1.mkv", FPS_desired=2,clip_len=16)
#     dataset = FrameDataset(
#         "/media/giancos/Football/SoccerNet/england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley/1.mkv", FPS_desired=2)
