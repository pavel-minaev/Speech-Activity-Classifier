from torch.utils.data import Dataset
import torchaudio
import torch
import torch.nn as nn

import pandas as pd


class AudioSileroMFCC(Dataset):
    def __init__(self, items_frame, labels_frame):
        assert type(items_frame) == pd.core.series.Series
        assert type(labels_frame) == pd.core.series.Series
        self.items = list(zip(list(items_frame), list(labels_frame)))
        self.length = len(self.items)

    def __getitem__(self, index):
        filename, label = self.items[index]
        audioTensor, rate = torchaudio.load(filename)
        MFCC_Tensor = torchaudio.transforms.MFCC(n_mfcc=70).forward(audioTensor)
        return (MFCC_Tensor, int(label))

    def __len__(self):
        return self.length


class MFCCSimpleAudioNet(nn.Module):
    def __init__(self):
        super(MFCCSimpleAudioNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.avgPool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(32 * 8 * 29, 3)

    def forward(self, x):
        x = self.features(x)
        x = self.avgPool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x.squeeze(1)


def create_data_loaders(file_path, batch_size, dataset_class):
    train_df = pd.read_csv(file_path, index_col='id')
    train_df.wav_path = train_df.wav_path.apply(lambda path: '../silero-audio-classifier/' + path)

    assert train_df.shape[0] > 12_000

    train_files = train_df.iloc[:10_000]
    validation_files = train_df.iloc[10_000:12_000]

    train_dataset = dataset_class(train_files.wav_path, train_files.target)
    valid_dataset = dataset_class(validation_files.wav_path, validation_files.target)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    print('loaders crated')

    return train_loader, valid_loader
