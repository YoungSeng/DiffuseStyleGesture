import logging
import pdb
import lmdb as lmdb
import torch
from torch.utils.data import Dataset
import pyarrow
import sys
import os
[sys.path.append(i) for i in ['.', '..']]
from data_loader.data_preprocessor import DataPreprocessor


class TrinityDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, subdivision_stride, pose_resampling_fps, model=None, device=torch.device('cuda:0')):
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.subdivision_stride = subdivision_stride
        self.skeleton_resampling_fps = pose_resampling_fps
        self.lang_model = None

        logging.info("Reading data '{}'...".format(lmdb_dir))
        if model is not None:
            if 'Long_' in model:
                preloaded_dir = lmdb_dir + '_cache_' + model.split('_')[-1]
            if 'WavLM' in model:
                preloaded_dir = lmdb_dir + '_cache_WavLM'
        else:
            preloaded_dir = lmdb_dir + '_cache'
        if not os.path.exists(preloaded_dir):
            data_sampler = DataPreprocessor(lmdb_dir, preloaded_dir, n_poses,
                                            subdivision_stride, pose_resampling_fps, device=device)
            data_sampler.run()
        else:
            logging.info('Found pre-loaded samples from {}'.format(preloaded_dir))

        # init lmdb
        # map_size = 1024 * 20  # in MB
        # map_size <<= 20  # in B
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)  # default 10485760
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)

            sample = pyarrow.deserialize(sample)
            # pose_seq, audio, styles, mfcc, wavlm, aux_info = sample
            pose_seq, styles, wavlm = sample

        # # normalize
        # std = np.clip(self.data_std, a_min=0.01, a_max=None)
        # pose_seq = (pose_seq - self.data_mean) / std

        # to tensors
        pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
        styles = torch.from_numpy(styles).float()
        # audio = torch.from_numpy(audio).float()
        # mfcc = torch.from_numpy(mfcc).float()
        wavlm = torch.from_numpy(wavlm).float()

        # return pose_seq, aux_info, styles, audio, mfcc, wavlm
        return pose_seq, styles, wavlm


if __name__ == '__main__':
    '''
    cd main/mydiffusion_zeggs
    python data_loader/lmdb_data_loader.py --config=./configs/DiffuseStyleGesture.yml --no_cuda 0 --gpu 0
    '''

    from configs.parse_args import parse_args
    import os
    import yaml
    from pprint import pprint
    from easydict import EasyDict
    from torch.utils.data import DataLoader

    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    # pprint(config)

    args = EasyDict(config)

    train_dataset = TrinityDataset(args.train_data_path,
                                   n_poses=args.n_poses,
                                   subdivision_stride=args.subdivision_stride,
                                   pose_resampling_fps=args.motion_resampling_framerate, model='WavLM', device=torch.device('cuda:0'))
    val_dataset = TrinityDataset(args.val_data_path,
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate, model='WavLM', device=torch.device('cuda:0'))
    train_loader = DataLoader(dataset=train_dataset, batch_size=128,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    print(len(train_loader))
    for batch_i, batch in enumerate(train_loader, 0):
        # target_vec, aux, style, audio, mfcc, wavlm = batch     # [128, 88, 1141], -,  [128, 6], [128, 70400], [128, 88, 13]
        target_vec, style, wavlm = batch
        print(batch_i)
        pdb.set_trace()
        # print(target_vec.shape, audio.shape)
