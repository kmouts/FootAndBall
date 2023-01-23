

# import torch
# # import pandas as pd


# class SoccerNetDataLoaderTensorFlow(torch.utils.data.Dataset):
    
#     def __init__(self, soccernet_dir, jsonGamesFile=None, transform=None, input="Video_LQ"):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """

#         if jsonGamesFile is None:
#             jsonGamesFile = Path(__file__).parent / "data/SoccerNetGames.json"

#         self.jsonGamesFile = jsonGamesFile

#         with open(jsonGamesFile, "r") as json_file:
#             dictionary = json.load(json_file)

#         self.games = []
#         for championship in dictionary:
#             for season in dictionary[championship]:
#                 for game in dictionary[championship][season]:

#                     self.games.append(os.path.join(championship, season, game))
                    

#         # self.landmarks_frame = pd.read_csv(csv_file)
#         self.soccernet_dir = soccernet_dir
#         self.transform = transform
#         self.input = input

#     def __len__(self):
#         return len(self.games)

#     def __getitem__(self, idx):
#         game = self.games[idx]

#         if self.input == "Video_LQ":
#             labels = scikit.io.vread(os.path.join(self.games[idx], "Labels.txt")
#             video1 = scikit.io.vread(os.path.join(self.games[idx], "1.mkv")
#             video2 = scikit.io.vread(os.path.join(self.games[idx], "2.mkv")

#             sample={'video1': video1, 'video2': video2,
#                     'labels': labels}

#             if self.transform:
#                 sample = self.transform(sample)

#         return sample
