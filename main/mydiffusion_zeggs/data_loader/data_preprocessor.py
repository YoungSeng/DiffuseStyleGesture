""" create data samples """
import pdb

import lmdb
import math
import numpy as np
import pyarrow


import torch
import torch.nn.functional as F

def wavlm_init(device=torch.device('cuda:1')):
    import sys
    [sys.path.append(i) for i in ['./WavLM']]
    from WavLM import WavLM, WavLMConfig
    wavlm_model_path = './WavLM/WavLM-Large.pt'
    # wavlm_model_path = '../../../My/process/WavLM-Base+.pt'
    # load the pre-trained checkpoints
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def wav2wavlm(model, wav_input_16khz, device=torch.device('cuda:1')):
    with torch.no_grad():
        wav_input_16khz = torch.from_numpy(wav_input_16khz).float()
        wav_input_16khz = wav_input_16khz.to(device).unsqueeze(0)
        rep = model.extract_features(wav_input_16khz)[0]
        rep = F.interpolate(rep.transpose(1, 2), size=88, align_corners=True, mode='linear').transpose(1, 2)
        return rep.squeeze().cpu().detach().data.cpu().numpy()


class DataPreprocessor:
    def __init__(self, clip_lmdb_dir, out_lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, device):
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps

        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']

        self.audio_sample_length = int(self.n_poses / self.skeleton_resampling_fps * 16000)

        # create db for samples
        map_size = 1024 * 1024 * 20  # in TB
        map_size <<= 20  # in B
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=map_size)
        self.n_out_samples = 0

        self.model = wavlm_init(device)
        self.device = device

    def run(self):
        src_txn = self.src_lmdb_env.begin(write=False)

        # sampling and normalization
        cursor = src_txn.cursor()
        for key, value in cursor:
            video = pyarrow.deserialize(value)
            vid = video['vid']
            clips = video['clips']
            for clip_idx, clip in enumerate(clips):
                self._sample_from_clip(vid, clip, self.device)

        # print stats
        with self.dst_lmdb_env.begin() as txn:
            print('no. of samples: ', txn.stat()['entries'])
        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()


    def _sample_from_clip(self, vid, clip, device):
        clip_skeleton = clip['poses']
        clip_audio_raw = clip['audio_raw']
        clip_styles_raw = clip['style_raw']
        clip_mfcc_raw = clip['mfcc_raw']

        # divide
        aux_info = []
        sample_skeletons_list = []
        sample_audio_list = []
        sample_codes_list = []
        sample_mfcc_list = []
        sample_wavlm_list = []

        MINLEN = min(len(clip_skeleton), int(len(clip_audio_raw) * 60 / 16000), len(clip_mfcc_raw))

        num_subdivision = math.floor(
            (MINLEN - self.n_poses)
            / self.subdivision_stride)  # floor((K - (N+M)) / S) + 1

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            fin_idx = start_idx + self.n_poses

            sample_skeletons = clip_skeleton[start_idx:fin_idx]
            sample_mfcc = clip_mfcc_raw[start_idx:fin_idx]
            subdivision_start_time = start_idx / self.skeleton_resampling_fps
            subdivision_end_time = fin_idx / self.skeleton_resampling_fps

            # raw audio
            audio_start = math.floor(start_idx / len(clip_skeleton) * len(clip_audio_raw))
            audio_end = audio_start + self.audio_sample_length
            sample_audio = clip_audio_raw[audio_start:audio_end]
            sample_wavlm = wav2wavlm(self.model, sample_audio, device=device)

            motion_info = {'vid': vid,
                           'start_frame_no': start_idx,
                           'end_frame_no': fin_idx,
                           'start_time': subdivision_start_time,
                           'end_time': subdivision_end_time}

            sample_skeletons_list.append(sample_skeletons)
            sample_mfcc_list.append(sample_mfcc)
            sample_wavlm_list.append(sample_wavlm)
            sample_audio_list.append(sample_audio)
            sample_codes_list.append(clip_styles_raw)
            aux_info.append(motion_info)

        # if len(sample_skeletons_list) > 0:
        #     with self.dst_lmdb_env.begin(write=True) as txn:
        #         for poses, audio, codes, mfcc, wavlm, aux in zip(sample_skeletons_list,
        #                                             sample_audio_list, sample_codes_list, sample_mfcc_list, sample_wavlm_list, aux_info):
        #             poses = np.asarray(poses)
        #
        #             # save
        #             k = '{:010}'.format(self.n_out_samples).encode('ascii')
        #             v = [poses, audio, codes, mfcc, wavlm, aux]
        #             v = pyarrow.serialize(v).to_buffer()
        #             txn.put(k, v)
        #             self.n_out_samples += 1

        if len(sample_skeletons_list) > 0:
            with self.dst_lmdb_env.begin(write=True) as txn:
                for poses, codes, wavlm in zip(sample_skeletons_list, sample_codes_list, sample_wavlm_list):
                    poses = np.asarray(poses)

                    # save
                    k = '{:010}'.format(self.n_out_samples).encode('ascii')
                    v = [poses, codes, wavlm]
                    v = pyarrow.serialize(v).to_buffer()
                    txn.put(k, v)
                    self.n_out_samples += 1

