import pdb
import yaml
from pprint import pprint
from easydict import EasyDict
import numpy as np
import torch
from configs.parse_args import parse_args
import math
from models.vqvae import VQVAE
import torch.nn as nn
from generate.generate import Generator_gru as Generator
import librosa
import os
import sys

[sys.path.append(i) for i in ['.', '..', '../process']]
from process.process_bvh import make_bvh_GENEA2020_BT
from process.bvh_to_position import bvh_to_npy
from process.visualize_bvh import visualize

args = parse_args()
mydevice = torch.device('cuda:' + args.gpu)


def main(args, audio_path, normalize=True, mode='position', codebook_model_path=None, end2end_model_path=None,
         save_path=None, prefix=None, max_frames=None):

    audio_raw, audio_sr = librosa.load(audio_path, mono=True, sr=16000, res_type='kaiser_fast')

    clip_length = audio_raw.shape[0]

    # divide into synthesize units and do synthesize
    unit_time = audio_sr * args.n_poses / args.motion_resampling_framerate       # 4 * 16000

    if clip_length < unit_time:
        num_subdivision = 1
    else:
        num_subdivision = math.ceil((clip_length - unit_time) / unit_time) + 1

    if max_frames is not None and num_subdivision >= (max_frames / args.motion_resampling_framerate) / args.n_poses:
        num_subdivision = int(max_frames / args.n_poses)       # 3600 / 60 / 4 = 15

    print('num_subdivision: {}, unit_time: {}, clip_length: {}'.format(num_subdivision, unit_time, clip_length))


    with torch.no_grad():
        if mode == 'position':
            model_VQVAE = VQVAE(args.VQVAE, 15 * 3)  # n_joints * n_chanels
        elif mode == 'rotation':
            model_VQVAE = VQVAE(args.VQVAE, 15 * 9)  # n_joints * n_chanels
        model_VQVAE = nn.DataParallel(model_VQVAE, device_ids=[eval(i) for i in config.no_cuda])
        model_VQVAE = model_VQVAE.to(mydevice)
        checkpoint = torch.load(codebook_model_path, map_location=torch.device('cpu'))
        model_VQVAE.load_state_dict(checkpoint['model_dict'])
        model_VQVAE = model_VQVAE.eval()

        model = Generator()
        model = nn.DataParallel(model, device_ids=[eval(i) for i in config.no_cuda])
        model = model.to(mydevice)
        checkpoint = torch.load(end2end_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_dict'])
        model = model.eval()

        result = []
        code = []

        for i in range(0, num_subdivision):
            start_time = i * unit_time

            # prepare pose input
            pose_start = math.floor(start_time)
            pose_end = pose_start + 64000
            in_audio = audio_raw[pose_start:pose_end]
            if len(in_audio) < 64000:
                in_audio = np.pad(in_audio, (0, 64000 - len(in_audio)), 'constant')
            in_audio = torch.from_numpy(in_audio).unsqueeze(0).to(mydevice)

            out_zs = model.module.sample(in_audio)

            code.append(out_zs[0].squeeze(0).data.cpu().numpy())

    out_code = np.vstack(code)
    # print(torch.from_numpy(out_code.flatten()).to(mydevice).unsqueeze(0).shape)
    pose_sample = model_VQVAE.module.decode([torch.from_numpy(out_code.flatten()).to(mydevice).unsqueeze(0)]).squeeze(0).data.cpu().numpy()
    result.append(pose_sample)

    out_poses = np.vstack(result)

    if normalize:
        data_mean = np.array(args.data_mean).squeeze()
        data_std = np.array(args.data_std).squeeze()
        std = np.clip(data_std, a_min=0.01, a_max=None)
        out_poses = np.multiply(out_poses, std) + data_mean
    print(out_poses.shape)
    print(out_code.shape)
    np.save(os.path.join(save_path, 'code' + prefix + '.npy'), out_code)
    np.save(os.path.join(save_path, 'generate' + prefix + '.npy'), out_poses)
    return out_poses, out_code


if __name__ == '__main__':
    '''
    cd codebook/
    python inference.py --config=./configs/codebook.yml --train --no_cuda 3 --gpu 3
    '''

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)

    # audio_path = "/mnt/nfs7/y50021900/My/tmp/TEST/TestSeq001.wav"
    audio_path = "../tmp/TEST/1_wayne_0_103_110.wav"
    mode = 'rotation'
    codebook_path = '../codebook/BEAT_output_60fps_rotation/train_codebook/' + "codebook_checkpoint_best.bin"
    end2end_path = "./BEAT_output_60fps_rotation_gru/train_end2end/end2end_checkpoint_080.bin"       #
    save_path = "../tmp/TEST/npy_position/"
    prefix = 'BEAT_gru'
    save_path = os.path.join(save_path, prefix)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    MAX_FRAMES = 60 * 60
    pipeline_path = '../process/resource/data_pipe_60_rotation.sav'

    out_poses, out_code = main(config, audio_path=audio_path, normalize=True, mode=mode, codebook_model_path=codebook_path,
                            end2end_model_path=end2end_path, save_path=save_path, prefix=prefix, max_frames=MAX_FRAMES)
    print('rotation npy to bvh...')
    # make_bvh_GENEA2020_BT(save_path, filename_prefix='GT', poses=poses, smoothing=False, pipeline_path=pipeline_path)
    make_bvh_GENEA2020_BT(save_path, prefix, out_poses, smoothing=False, pipeline_path=pipeline_path)
    print('bvh to position npy...')
    bvh_path_generated = os.path.join(save_path, prefix + '_generated.bvh')
    bvh_to_npy(bvh_path_generated, save_path)
    print('visualize code...')
    npy_generated = np.load(os.path.join(save_path, prefix + '_generated.npy'))
    out_video = os.path.join(save_path, prefix + '_generated.mp4')
    visualize(npy_generated.reshape((npy_generated.shape[0], -1, 3)), out_video, out_code.flatten(), 'upper')
