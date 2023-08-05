import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
import pdb
import os


speaker_id_dict = {
    2: 0,
    10: 1,
}


class SpeechGestureDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, motion_dim, style_dim, sequence_length=30*5, npy_root="../../process", 
                 version='v0', dataset='BEAT'):
        self.h5 = h5py.File(h5file, "r")
        self.len = len(self.h5.keys())
        self.motion_dim = motion_dim
        self.style_dim = style_dim
        self.version = version
        
        gesture_mean = np.load(os.path.join(npy_root, "gesture_" + dataset + "_mean_" + self.version + ".npy"))
        gesture_std = np.load(os.path.join(npy_root, "gesture_" + dataset + "_std_" + self.version + ".npy"))

        self.id = [speaker_id_dict[int(self.h5[str(i)]["speaker_id"][:][0])] for i in range(len(self.h5.keys()))]
        self.audio = [self.h5[str(i)]["audio"][:] for i in range(len(self.h5.keys()))]
        self.text = [self.h5[str(i)]["text"][:] for i in range(len(self.h5.keys()))]
        self.gesture = [(self.h5[str(i)]["gesture"][:] - gesture_mean) / gesture_std for i in range(len(self.h5.keys()))]
        self.h5.close()
        self.sequence_length = sequence_length
        if "v0" in self.version:
            self.gesture_vel = [np.concatenate((np.zeros([1, self.motion_dim]), i[1:] - i[:-1]), axis=0) for i in self.gesture]
            self.gesture_acc = [np.concatenate((np.zeros([1, self.motion_dim]), i[1:] - i[:-1]), axis=0) for i in self.gesture_vel]
        print("Total clips:", len(self.gesture))
        self.segment_length = sequence_length

    def __len__(self):
        return len(self.gesture)

    def __getitem__(self, idx):
        total_frame_len = self.audio[idx].shape[0]
        start_frame = np.random.randint(0, total_frame_len - self.segment_length)
        end_frame = start_frame + self.segment_length
        audio = self.audio[idx][start_frame:end_frame]
        text = self.text[idx][start_frame:end_frame]
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        posrat = self.gesture[idx][start_frame:end_frame]
        
        # if "v0" in self.version:
        #     vel = self.gesture_vel[idx][start_frame:end_frame]
        #     acc = self.gesture_acc[idx][start_frame:end_frame]
        #     gesture = np.concatenate((posrat, vel, acc), axis=-1)
        # else:
        #     gesture = posrat
        vel = self.gesture_vel[idx][start_frame:end_frame]
        acc = self.gesture_acc[idx][start_frame:end_frame]
        gesture = np.concatenate((posrat, vel, acc), axis=-1)
        # gesture = posrat
        
        gesture = torch.FloatTensor(gesture)
        speaker = np.zeros([self.style_dim])
        # speaker[0] = 1      # dummy speaker
        speaker[self.id[idx]] = 1
        speaker = torch.FloatTensor(speaker)
        return textaudio, gesture, speaker


class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        while True:
            yield np.random.randint(self.min_id, self.max_id)


class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        return iter(range(self.min_id, self.max_id))


if __name__ == '__main__':
    '''
    cd ./BEAT-main/mydiffusion_beat/data_loader
    python h5_data_loader.py
    '''
    # Get data, data loaders and collate function ready
    print("Loading dataset into memory ...")
    trn_dataset = SpeechGestureDataset("../../process/speaker_2_10_v0.h5", motion_dim=684, style_dim=2)

    train_loader = DataLoader(trn_dataset, num_workers=4,
                              sampler=RandomSampler(0, len(trn_dataset)),
                              batch_size=128,
                              pin_memory=True,
                              drop_last=False)

    for batch_i, batch in enumerate(train_loader, 0):
        textaudio, gesture, speaker = batch     # (128, 150, 1435), (128, 150, 744), (128, 17)
        print(batch_i)
        pdb.set_trace()
