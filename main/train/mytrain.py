import pdb
import sys
[sys.path.append(i) for i in ['.', '..']]

from model.mdm import MDM
from utils.model_util import create_gaussian_diffusion

# from data_loaders.get_data import get_dataset_loader
from train.training_loop import TrainLoop

import torch
import os
import json


device = torch.device('cuda:2')
n_frames = 240
n_pose_dims = 135
n_audio_dim = 32

# n_frames = 240
# n_pose_dims = 251


def create_model_and_diffusion():
    model = MDM(modeltype='', njoints=n_pose_dims, nfeats=1, translation=True, pose_rep='rot6d', glob=True,
                glob_rot=True, cond_mode = 'text', clip_version = 'ViT-B/32', action_emb = 'tensor')
    diffusion = create_gaussian_diffusion()
    return model, diffusion


if __name__ == '__main__':
    '''
    python train/mytrain.py --overwrite --save_dir save/mydebug --dataset kit --device 1
    '''
    # modify data/dataset.py

    model, diffusion = create_model_and_diffusion()
    model.to(device)
    # model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))


    from utils.parser_util import train_args
    from utils.fixseed import fixseed

    args = train_args()
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    # print("creating data loader...")
    # data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames)

    # print(iter(data).next()[1]['y'].keys())
    print("Training...")
    TrainLoop(args, model, diffusion, device).run_loop()

