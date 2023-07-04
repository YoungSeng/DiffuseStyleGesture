import sys
[sys.path.append(i) for i in ['.', '..', '../process', '../model', '../../ubisoft-laforge-ZeroEGGS-main', '../../ubisoft-laforge-ZeroEGGS-main/ZEGGS']]
from model.mdm import MDM
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
import subprocess
import os
from datetime import datetime
from mfcc import MFCC
import librosa
import numpy as np
import yaml
from pprint import pprint
import torch
import torch.nn.functional as F
from easydict import EasyDict
import math
from process_zeggs_bvh import pose2bvh, quat      # '../process'
import argparse

style2onehot = {
'Happy':[1, 0, 0, 0, 0, 0],
'Sad':[0, 1, 0, 0, 0, 0],
'Neutral':[0, 0, 1, 0, 0, 0],
'Old':[0, 0, 0, 1, 0, 0],
'Angry':[0, 0, 0, 0, 1, 0],
'Relaxed':[0, 0, 0, 0, 0, 1],
}


def wavlm_init(device=torch.device('cuda:2')):
    import sys
    [sys.path.append(i) for i in ['./WavLM']]
    from WavLM import WavLM, WavLMConfig
    wavlm_model_path = './WavLM/WavLM-Large.pt'
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))     # load the pre-trained checkpoints
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def wav2wavlm(model, wav_input_16khz, device=torch.device('cuda:2')):
    wav_input_16khz = wav_input_16khz.to(device)
    rep = model.extract_features(wav_input_16khz)[0]
    rep = F.interpolate(rep.transpose(1, 2), size=88, align_corners=True, mode='linear').transpose(1, 2)
    return rep


def create_model_and_diffusion(args):
    model = MDM(modeltype='', njoints=1141, nfeats=1, translation=True, pose_rep='rot6d', glob=True,
                glob_rot=True, cond_mode = 'cross_local_attention3_style1', clip_version = 'ViT-B/32', action_emb = 'tensor', audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=256, n_seed=8)        # trans_enc, trans_dec, gru, mytrans_enc
    diffusion = create_gaussian_diffusion()
    return model, diffusion


def inference_mfcc(args, mfcc, sample_fn, model, n_frames=0, smoothing=False, SG_filter=False, minibatch=False, skip_timesteps=0, n_seed=8, style=None, seed=123456, smooth_foot=False):

    torch.manual_seed(seed)

    if n_frames == 0:
        n_frames = mfcc.shape[0]
    if minibatch:
        stride_poses = args.n_poses - n_seed
        if n_frames < stride_poses:
            num_subdivision = 1
        else:
            num_subdivision = math.floor(n_frames / stride_poses)
            n_frames = num_subdivision * stride_poses
            print(
                '{}, {}, {}'.format(num_subdivision, stride_poses, n_frames))
    mfcc = mfcc[:n_frames]

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1).to(mydevice)
    model_kwargs_['y']['style'] = torch.as_tensor([style]).float().to(mydevice)
    model_kwargs_['y']['mask_local'] = torch.ones(1, args.n_poses).bool().to(mydevice)

    # tmp_mfcc = torch.from_numpy(np.load('10_kieks_0_9_16.npz')['mfcc'][:n_frames]).to(torch.float32).unsqueeze(0).to(mydevice)
    # model_kwargs_['y']['audio'] = tmp_mfcc.permute(1, 0, 2)

    if minibatch:
        audio_reshape = torch.from_numpy(mfcc).to(torch.float32).reshape(num_subdivision, stride_poses, -1).to(mydevice).permute(1, 0, 2)       # mfcc[:, :-2]
        shape_ = (1, model.njoints, model.nfeats, args.n_poses)
        out_list = []
        for i in range(0, num_subdivision):
            print(i, num_subdivision)
            model_kwargs_['y']['audio'] = audio_reshape[:, i:i + 1, :]
            if i == 0:
                if n_seed != 0:
                    pad_zeros = torch.zeros([n_seed, 1, 13]).to(mydevice)        # mfcc dims are 13
                    model_kwargs_['y']['audio'] = torch.cat((pad_zeros, model_kwargs_['y']['audio']), 0)
                    model_kwargs_['y']['seed'] = torch.zeros([1, 1141, 1, n_seed]).to(mydevice)
            else:
                if n_seed != 0:
                    pad_audio = audio_reshape[-n_seed:, i - 1:i, :]
                    model_kwargs_['y']['audio'] = torch.cat((pad_audio, model_kwargs_['y']['audio']), 0)
                    model_kwargs_['y']['seed'] = out_list[-1][..., -n_seed:].to(mydevice)

            sample = sample_fn(
                model,
                shape_,
                clip_denoised=False,
                model_kwargs=model_kwargs_,
                skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,  # None, torch.randn(*shape_, device=mydevice)
                const_noise=False,
            )
            # smoothing motion transition
            if len(out_list) > 0 and n_seed != 0:
                last_poses = out_list[-1][..., -n_seed:]        # # (1, model.njoints, 1, n_seed)
                out_list[-1] = out_list[-1][..., :-n_seed]  # delete last 4 frames
                if smoothing:
                    # Extract predictions
                    last_poses_root_pos = last_poses[:, 0:3]        # (1, 3, 1, 8)
                    # last_poses_root_rot = last_poses[:, 3:7]
                    # last_poses_root_vel = last_poses[:, 7:10]
                    # last_poses_root_vrt = last_poses[:, 10:13]
                    next_poses_root_pos = sample[:, 0:3]        # (1, 3, 1, 88)
                    # next_poses_root_rot = sample[:, 3:7]
                    # next_poses_root_vel = sample[:, 7:10]
                    # next_poses_root_vrt = sample[:, 10:13]
                    root_pos = last_poses_root_pos[..., 0]      # (1, 3, 1)
                    predict_pos = next_poses_root_pos[..., 0]
                    delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
                    sample[:, 0:3] = sample[:, 0:3] - delta_pos

                if smooth_foot:
                    njoints = 75
                    length = n_seed
                    last_poses_lpos = last_poses[:, 13 + njoints * 0: 13 + njoints * 3].reshape([length, njoints, 3])
                    last_poses_LeftToeBase = last_poses_lpos[0, -4]
                    last_poses_RightToeBase = last_poses_lpos[0, -11]

                    next_poses_lpos = sample[:, 13 + njoints * 0: 13 + njoints * 3].reshape([args.n_poses, njoints, 3])
                    next_poses_LeftToeBase = next_poses_lpos[0, -4]
                    next_poses_RightToeBase = next_poses_lpos[0, -11]

                    delta_poses_LeftToeBase = (next_poses_LeftToeBase - last_poses_LeftToeBase)
                    delta_poses_RightToeBase = (next_poses_RightToeBase - last_poses_RightToeBase)

                    next_poses_lpos[:, -4] = (next_poses_lpos[:, -4] - delta_poses_LeftToeBase)
                    next_poses_lpos[:, -11] = (next_poses_lpos[:, -11] - delta_poses_RightToeBase)
                    sample[:, 13 + njoints * 0: 13 + njoints * 3] = next_poses_lpos.reshape(1, -1, 1, args.n_poses)

                for j in range(len(last_poses)):
                    n = len(last_poses)
                    prev = last_poses[..., j]
                    next = sample[..., j]
                    sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
            out_list.append(sample)

        if n_seed != 0:
            out_list[-1] = out_list[-1][..., :-n_seed]
            out_list = [i.detach().data.cpu().numpy() for i in out_list]
            out_dir_vec = np.vstack(out_list)
            sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)
            sampled_seq = sampled_seq[:, n_seed:]
        else:
            out_list = [i.detach().data.cpu().numpy() for i in out_list]
            out_dir_vec = np.vstack(out_list)
            sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)
    else:
        model_kwargs_['y']['audio'] = torch.from_numpy(mfcc).to(torch.float32).unsqueeze(0).to(mydevice).permute(1, 0, 2)
        shape_ = (batch_size, model.njoints, model.nfeats, n_frames)
        model_kwargs_['y']['seed'] = torch.zeros([1, 1141, 1, n_seed]).to(mydevice)
        sample = sample_fn(
            model,
            shape_,
            clip_denoised=False,
            model_kwargs=model_kwargs_,
            skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,  # None, torch.randn(*shape_, device=mydevice)
            const_noise=False,
        )
        out_dir_vec = sample.data.cpu().numpy()
        sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)

    data_mean_ = np.load("../../ubisoft-laforge-ZeroEGGS-main/data/processed_v1/processed/mean.npz")['mean'].squeeze()
    data_std_ = np.load("../../ubisoft-laforge-ZeroEGGS-main/data/processed_v1/processed/std.npz")['std'].squeeze()

    data_mean = np.array(data_mean_).squeeze()
    data_std = np.array(data_std_).squeeze()
    std = np.clip(data_std, a_min=0.01, a_max=None)
    out_poses = np.multiply(sampled_seq[0], std) + data_mean
    print(out_poses.shape)
    pipeline_path = '../../../My/process/resource/data_pipe_20_rotation.sav'
    prefix = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    if smoothing: prefix += '_smoothing'
    if smooth_foot: prefix += 'smoothfoot'
    if SG_filter: prefix += '_SG'
    if minibatch: prefix += '_minibatch'
    prefix += '_%s' % (n_frames)
    prefix += '_' + str(style)
    prefix += '_' + str(seed)
    if minibatch:
        pose2bvh(out_poses, os.path.join(save_dir, prefix + '.bvh'), length=n_frames - n_seed, smoothing=SG_filter)
    else:
        pose2bvh(out_poses, os.path.join(save_dir, prefix + '.bvh'), length=n_frames, smoothing=SG_filter)


def inference(args, wavlm_model, audio, sample_fn, model, n_frames=0, smoothing=False, SG_filter=False, minibatch=False, skip_timesteps=0, n_seed=8, style=None, seed=123456):

    torch.manual_seed(seed)

    if n_frames == 0:
        n_frames = audio.shape[0] * 20 // 16000
    if minibatch:
        stride_poses = args.n_poses - n_seed
        if n_frames < stride_poses:
            num_subdivision = 1
        else:
            num_subdivision = math.floor(n_frames / stride_poses)
            n_frames = num_subdivision * stride_poses
            print(
                '{}, {}, {}'.format(num_subdivision, stride_poses, n_frames))
    audio = audio[:int(n_frames * 16000 / 20)]

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1).to(mydevice)
    model_kwargs_['y']['style'] = torch.as_tensor([style]).float().to(mydevice)
    model_kwargs_['y']['mask_local'] = torch.ones(1, args.n_poses).bool().to(mydevice)

    if minibatch:
        audio_reshape = torch.from_numpy(audio).to(torch.float32).reshape(num_subdivision, int(stride_poses * 16000 / 20)).to(mydevice).transpose(0, 1)       # mfcc[:, :-2]
        shape_ = (1, model.njoints, model.nfeats, args.n_poses)
        out_list = []
        for i in range(0, num_subdivision):
            print(i, num_subdivision)
            model_kwargs_['y']['audio'] = audio_reshape[:, i:i + 1]

            if i == 0:
                if n_seed != 0:
                    pad_zeros = torch.zeros([int(n_seed * 16000 / 20), 1]).to(mydevice)        # wavlm dims are 1024
                    model_kwargs_['y']['audio'] = torch.cat((pad_zeros, model_kwargs_['y']['audio']), 0)
                    model_kwargs_['y']['seed'] = torch.zeros([1, 1141, 1, n_seed]).to(mydevice)
            else:
                if n_seed != 0:
                    pad_audio = audio_reshape[-int(n_seed * 16000 / 20):, i - 1:i]
                    model_kwargs_['y']['audio'] = torch.cat((pad_audio, model_kwargs_['y']['audio']), 0)
                    model_kwargs_['y']['seed'] = out_list[-1][..., -n_seed:].to(mydevice)

            model_kwargs_['y']['audio'] = wav2wavlm(wavlm_model, model_kwargs_['y']['audio'].transpose(0, 1), mydevice)

            sample = sample_fn(
                model,
                shape_,
                clip_denoised=False,
                model_kwargs=model_kwargs_,
                skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,  # None, torch.randn(*shape_, device=mydevice)
                const_noise=False,
            )
            # smoothing motion transition
            if len(out_list) > 0 and n_seed != 0:
                last_poses = out_list[-1][..., -n_seed:]        # # (1, model.njoints, 1, n_seed)
                out_list[-1] = out_list[-1][..., :-n_seed]  # delete last 4 frames
                if smoothing:
                    # Extract predictions
                    last_poses_root_pos = last_poses[:, 0:3]        # (1, 3, 1, 8)
                    # last_poses_root_rot = last_poses[:, 3:7]
                    # last_poses_root_vel = last_poses[:, 7:10]
                    # last_poses_root_vrt = last_poses[:, 10:13]
                    next_poses_root_pos = sample[:, 0:3]        # (1, 3, 1, 88)
                    # next_poses_root_rot = sample[:, 3:7]
                    # next_poses_root_vel = sample[:, 7:10]
                    # next_poses_root_vrt = sample[:, 10:13]
                    root_pos = last_poses_root_pos[..., 0]      # (1, 3, 1)
                    predict_pos = next_poses_root_pos[..., 0]
                    delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
                    sample[:, 0:3] = sample[:, 0:3] - delta_pos

                for j in range(len(last_poses)):
                    n = len(last_poses)
                    prev = last_poses[..., j]
                    next = sample[..., j]
                    sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
            out_list.append(sample)

        if n_seed != 0:
            out_list[-1] = out_list[-1][..., :-n_seed]
            out_list = [i.detach().data.cpu().numpy() for i in out_list]
            out_dir_vec = np.vstack(out_list)
            sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)
            sampled_seq = sampled_seq[:, n_seed:]
        else:
            out_list = [i.detach().data.cpu().numpy() for i in out_list]
            out_dir_vec = np.vstack(out_list)
            sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)
    else:
        model_kwargs_['y']['audio'] = torch.from_numpy(mfcc).to(torch.float32).unsqueeze(0).to(mydevice).permute(1, 0, 2)
        shape_ = (batch_size, model.njoints, model.nfeats, n_frames)
        model_kwargs_['y']['seed'] = torch.zeros([1, 1141, 1, n_seed]).to(mydevice)
        sample = sample_fn(
            model,
            shape_,
            clip_denoised=False,
            model_kwargs=model_kwargs_,
            skip_timesteps=skip_timesteps,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,  # None, torch.randn(*shape_, device=mydevice)
            const_noise=False,
        )
        out_dir_vec = sample.data.cpu().numpy()
        sampled_seq = out_dir_vec.squeeze(2).transpose(0, 2, 1).reshape(batch_size, n_frames, model.njoints)

    data_mean_ = np.load("../../ubisoft-laforge-ZeroEGGS-main/data/processed_v1/processed/mean.npz")['mean'].squeeze()
    data_std_ = np.load("../../ubisoft-laforge-ZeroEGGS-main/data/processed_v1/processed/std.npz")['std'].squeeze()

    data_mean = np.array(data_mean_).squeeze()
    data_std = np.array(data_std_).squeeze()
    std = np.clip(data_std, a_min=0.01, a_max=None)
    out_poses = np.multiply(sampled_seq[0], std) + data_mean
    print(out_poses.shape)
    prefix = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    if smoothing: prefix += '_smoothing'
    if SG_filter: prefix += '_SG'
    if minibatch: prefix += '_minibatch'
    prefix += '_%s' % (n_frames)
    prefix += '_' + str(style)
    prefix += '_' + str(seed)
    if minibatch:
        pose2bvh(out_poses, os.path.join(save_dir, prefix + '.bvh'), length=n_frames - n_seed, smoothing=SG_filter)
    else:
        pose2bvh(out_poses, os.path.join(save_dir, prefix + '.bvh'), length=n_frames, smoothing=SG_filter)


def main(args, save_dir, model_path, audio_path=None, mfcc_path=None, audiowavlm_path=None, max_len=0):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if audiowavlm_path != None:
        mfcc, fs = librosa.load(audiowavlm_path, sr=16000)

    elif audio_path != None and mfcc_path == None:
        # normalize_audio
        audio_name = audio_path.split('/')[-1]
        print('normalize audio: ' + audio_name)
        normalize_wav_path = os.path.join(save_dir, 'normalize_' + audio_name)
        cmd = ['ffmpeg-normalize', audio_path, '-o', normalize_wav_path, '-ar', '16000']
        subprocess.call(cmd)

        # MFCC, https://github.com/supasorn/synthesizing_obama_network_training
        print('extract MFCC...')
        obj = MFCC(frate=20)
        wav, fs = librosa.load(normalize_wav_path, sr=16000)
        mfcc = obj.sig2s2mfc_energy(wav, None)
        print(mfcc[:, :-2].shape)  # -1 -> -2      # (502, 13)
        np.savez_compressed(os.path.join(save_dir, audio_name[:-4] + '.npz'), mfcc=mfcc[:, :-2])

    elif mfcc_path != None and audio_path == None:
        mfcc = np.load(mfcc_path)['mfcc']

    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(mydevice)
    model.eval()

    sample_fn = diffusion.p_sample_loop     # predict x_start

    style = style2onehot[audiowavlm_path.split('/')[-1].split('_')[1]]
    # style = [0, 0, 1, 0, 0, 0]
    # style = style2onehot['Neutral']
    print(style)

    wavlm_model = wavlm_init(mydevice)
    inference(args, wavlm_model, mfcc, sample_fn, model, n_frames=max_len, smoothing=True, SG_filter=True, minibatch=True, skip_timesteps=0, style=style, seed=123456)      # style2onehot['Happy']


if __name__ == '__main__':
    '''
    cd /ceph/hdd/yangsc21/Python/DSG/
    '''

    # audio_path = '../../../My/Test_audio/Example1/ZeroEGGS_cut.wav'
    # mfcc_path = "../../ubisoft-laforge-ZeroEGGS-main/data/processed_v1/processed/valid/mfcc/015_Happy_4_mirror_x_1_0.npz"       # 010_Sad_4_x_1_0.npz
    # audiowavlm_path = "./015_Happy_4_x_1_0.wav"

    # prefix = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    # save_dir = 'sample_' + prefix
    save_dir = 'sample_dir'

    parser = argparse.ArgumentParser(description='DiffuseStyleGesture')
    parser.add_argument('--config', default='./configs/DiffuseStyleGesture.yml')
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--no_cuda', type=list, default=['2'])
    parser.add_argument('--model_path', type=str, default='./model000450000.pt')
    parser.add_argument('--audiowavlm_path', type=str, default='')
    parser.add_argument('--max_len', type=int, default=0)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)
    mydevice = torch.device('cuda:' + config.gpu)
    torch.cuda.set_device(int(config.gpu))

    batch_size = 1

    main(config, save_dir, config.model_path, audio_path=None, mfcc_path=None, audiowavlm_path=config.audiowavlm_path, max_len=config.max_len)

