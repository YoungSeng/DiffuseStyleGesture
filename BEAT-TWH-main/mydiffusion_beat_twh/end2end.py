import pdb
import logging
logging.getLogger().setLevel(logging.INFO)
from torch.utils.data import DataLoader
from data_loader.h5_data_loader import SpeechGestureDataset, RandomSampler, SequentialSampler
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
    model = MDM(modeltype='', njoints=args.njoints, nfeats=1, cond_mode=args.cond_mode, audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=args.latent_dim, n_seed=args.n_seed, cond_mask_prob=args.cond_mask_prob, device=device_name,
                style_dim=args.style_dim, source_audio_dim=args.audio_feature_dim, 
                audio_feat_dim_latent=args.audio_feat_dim_latent)
    diffusion = create_gaussian_diffusion()
    return model, diffusion


def main(args):
    # Get data, data loaders and collate function ready
    print("Loading dataset into memory ...")
    trn_dataset = SpeechGestureDataset(args.h5file, motion_dim=args.motion_dim, style_dim=args.style_dim,
                                       sequence_length=args.n_poses, npy_root="../process", 
                                       version=args.version, dataset=args.dataset)        # debug

    train_loader = DataLoader(trn_dataset, num_workers=args.num_workers,
                              sampler=RandomSampler(0, len(trn_dataset)),
                              batch_size=args.batch_size,
                              pin_memory=True,
                              drop_last=False)

    model, diffusion = create_model_and_diffusion(args)
    model.to(mydevice)
    TrainLoop(args, model, diffusion, mydevice, data=train_loader).run_loop()


if __name__ == '__main__':
    '''
    cd ./BEAT-main/mydiffusion_beat/
    python end2end.py --config=./configs/DiffuseStyleGesture.yml --gpu 0
    '''

    args = parse_args()
    device_name = 'cuda:' + args.gpu
    mydevice = torch.device(device_name)
    torch.cuda.set_device(int(args.gpu))
    args.no_cuda = args.gpu

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    # pprint(config)

    config = EasyDict(config)

    print(config.name)
    assert config.name in ['DiffuseStyleGesture', 'DiffuseStyleGesture+', 'DiffuseStyleGesture++']
    if config.name == 'DiffuseStyleGesture++':
        config.cond_mode = 'cross_local_attention5_style1'
    elif config.name == 'DiffuseStyleGesture+':
        config.cond_mode = 'cross_local_attention4_style1'
    elif config.name == 'DiffuseStyleGesture':
        config.cond_mode = 'cross_local_attention3_style1'
        
    config.save_dir = "./" + config.dataset + "_mymodel4_512" + '_' + config.version
    if config.suffix != "":
        config.save_dir = config.save_dir + '_' + config.suffix
    print('model save path: ', config.save_dir, '   version:', config.version)
    
    if config.dataset == 'BEAT':
        config.style_dim = 2
        config.audio_feature_dim = 1434
        if 'v0' in config.version:
            config.motion_dim = 684
            config.njoints = 2052
        elif 'v2' in config.version:
            config.motion_dim = 1141
            config.njoints = 1141
    elif config.dataset == 'TWH':
        if 'v0' in config.version:
            config.motion_dim = 744
            config.njoints = 2232
            config.latent_dim = 512
            config.audio_feat_dim_latent = 128
            config.style_dim = 17
            config.audio_feature_dim = 1435     # with laugh
    else:
        raise NotImplementedError
    config.h5file = '../process/' + config.dataset + '_' + config.version + '.h5'
    main(config)
