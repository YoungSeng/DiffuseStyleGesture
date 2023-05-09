import pdb
import logging
logging.getLogger().setLevel(logging.INFO)
from torch.utils.data import DataLoader
from data_loader.lmdb_data_loader import TrinityDataset
import torch
import yaml
from pprint import pprint
from easydict import EasyDict
from configs.parse_args import parse_args
import os
import sys
[sys.path.append(i) for i in ['.', '..', '../model', '../train']]
from utils.model_util import create_gaussian_diffusion
from training_loop import TrainLoop
from model.mdm import MDM


def create_model_and_diffusion(args):
    model = MDM(modeltype='', njoints=1141, nfeats=1, cond_mode = 'cross_local_attention3_style1', action_emb = 'tensor', audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=256, n_seed=8, cond_mask_prob=0.1)
    diffusion = create_gaussian_diffusion()
    return model, diffusion


def main(args, device):
    # dataset
    train_dataset = TrinityDataset(args.train_data_path,
                                   n_poses=args.n_poses,
                                   subdivision_stride=args.subdivision_stride,
                                   pose_resampling_fps=args.motion_resampling_framerate, model='WavLM', device=device)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=args.loader_workers, pin_memory=True)

    val_dataset = TrinityDataset(args.val_data_path,
                                       n_poses=args.n_poses,
                                       subdivision_stride=args.subdivision_stride,
                                       pose_resampling_fps=args.motion_resampling_framerate, model='WavLM', device=device)
    test_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=True, num_workers=args.loader_workers, pin_memory=False)

    logging.info('len of train loader:{}, len of test loader:{}'.format(len(train_loader), len(test_loader)))

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    model, diffusion = create_model_and_diffusion(args)
    model.to(mydevice)
    TrainLoop(args, model, diffusion, mydevice, data=train_loader).run_loop()


if __name__ == '__main__':
    '''
    cd mydiffusion_zeggs/
    '''

    args = parse_args()
    mydevice = torch.device('cuda:' + args.gpu)
    torch.cuda.set_device(int(args.gpu))

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)

    main(config, mydevice)
