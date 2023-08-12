import pdb
import sys
[sys.path.append(i) for i in ['.', '..', '../process', '../model']]
from model.mdm import MDM
from utils.model_util import create_gaussian_diffusion, load_model_wo_clip
import subprocess
import os
from datetime import datetime
import copy
import librosa
import numpy as np
import yaml
from pprint import pprint
import torch
import torch.nn.functional as F
from easydict import EasyDict
import math
from process_BEAT_bvh import wav2wavlm, pose2bvh, pose2bvh_bugfix
from process_TWH_bvh import pose2bvh as pose2bvh_twh
from process_TWH_bvh import wavlm_init, load_metadata
import argparse


speaker_id_dict = {
    2: 0,
    10: 1,
}

id_speaker_dict = {
    0: 2,
    1: 10,
}


def create_model_and_diffusion(args):
    model = MDM(modeltype='', njoints=args.njoints, nfeats=1, cond_mode=config.cond_mode, audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=args.latent_dim, n_seed=args.n_seed, cond_mask_prob=args.cond_mask_prob, device=device_name,
                style_dim=args.style_dim, source_audio_dim=args.audio_feature_dim,
                audio_feat_dim_latent=args.audio_feat_dim_latent)
    diffusion = create_gaussian_diffusion()
    return model, diffusion


def inference(args, save_dir, prefix, textaudio, sample_fn, model, n_frames=0, smoothing=False, skip_timesteps=0, style=None, seed=123456, dataset='BEAT'):

    torch.manual_seed(seed)
    if dataset == 'BEAT':
        speaker = id_speaker_dict[np.argwhere(style == 1)[0][0]]
        assert speaker in speaker_id_dict.keys()
    elif dataset == 'TWH':
        speaker = np.where(style == np.max(style))[0][0]
    if n_frames == 0:
        n_frames = textaudio.shape[0]
    else:
        textaudio = textaudio[:n_frames]
    real_n_frames = copy.deepcopy(n_frames)     # 1830
    stride_poses = args.n_poses - args.n_seed
    if n_frames < stride_poses:
        num_subdivision = 1
        n_frames = stride_poses
    else:
        num_subdivision = math.ceil(n_frames / stride_poses)
        n_frames = num_subdivision * stride_poses
        print('real_n_frames: {}, num_subdivision: {}, stride_poses: {}, n_frames: {}, speaker_id: {}'.format(real_n_frames, num_subdivision, stride_poses, n_frames, np.where(style==np.max(style))[0][0]))

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, args.n_poses]) < 1).to(mydevice)
    model_kwargs_['y']['style'] = torch.as_tensor([style]).float().to(mydevice)
    model_kwargs_['y']['mask_local'] = torch.ones(1, args.n_poses).bool().to(mydevice)
    
    textaudio_pad = torch.zeros([n_frames - real_n_frames, args.audio_feature_dim]).to(mydevice)
    textaudio = torch.cat((textaudio, textaudio_pad), 0)
    audio_reshape = textaudio.reshape(num_subdivision, stride_poses, args.audio_feature_dim).transpose(0, 1)

    if dataset == 'BEAT':
        data_mean_ = np.load("../process/gesture_BEAT_mean_" + args.version + ".npy")
        data_std_ = np.load("../process/gesture_BEAT_std_" + args.version + ".npy")
    elif dataset == 'TWH':
        data_mean_ = np.load("../process/gesture_TWH_mean_v0" + ".npy")
        data_std_ = np.load("../process/gesture_TWH_std_v0" + ".npy")

    data_mean = np.array(data_mean_)
    data_std = np.array(data_std_)
    # std = np.clip(data_std, a_min=0.01, a_max=None)
    if args.name == 'DiffuseStyleGesture++':
        gesture_flag1 = np.load("../../BEAT_dataset/processed/" + 'gesture_BEAT' + "/2_scott_0_1_1.npy")[:args.n_seed + 2]
        gesture_flag1 = (gesture_flag1 - data_mean) / data_std
        gesture_flag1_vel = gesture_flag1[1:] - gesture_flag1[:-1]
        gesture_flag1_acc = gesture_flag1_vel[1:] - gesture_flag1_vel[:-1]
        gesture_flag1_ = np.concatenate((gesture_flag1[2:], gesture_flag1_vel[1:], gesture_flag1_acc), axis=1)  # (args.n_seed, args.njoints)
        gesture_flag1_ = torch.from_numpy(gesture_flag1_).float().transpose(0, 1).unsqueeze(0).to(mydevice)
        gesture_flag1_ = gesture_flag1_.unsqueeze(2)
        model_kwargs_['y']['seed_last'] = gesture_flag1_

    
    shape_ = (1, model.njoints, model.nfeats, args.n_poses)
    out_list = []
    for i in range(0, num_subdivision):
        print(i, num_subdivision)
        model_kwargs_['y']['audio'] = audio_reshape[:, i:i + 1]
        if i == 0:
            if args.name == 'DiffuseStyleGesture':
                pad_zeros = torch.zeros([args.n_seed, 1, args.audio_feature_dim]).to(mydevice)
                model_kwargs_['y']['audio'] = torch.cat((pad_zeros, model_kwargs_['y']['audio']), 0).transpose(0, 1)      # attention 3
            elif args.name == 'DiffuseStyleGesture+':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)       # attention 4
            elif args.name == 'DiffuseStyleGesture++':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'][:-args.n_seed, ...].transpose(0, 1)       # attention 5
                
            # model_kwargs_['y']['seed'] = torch.zeros([1, args.njoints, 1, args.n_seed]).to(mydevice)

            if dataset == 'BEAT':

                if speaker == 2:
                    seed_gesture = np.load("../../BEAT_dataset/processed/" + 'gesture_BEAT' + "/2_scott_0_1_1.npy")[:args.n_seed + 2]         # any speaker, here we only use seed pose of 2_scott_0_1_1.npy
                elif speaker == 10:
                    seed_gesture = np.load("../../BEAT_dataset/processed/" + 'gesture_BEAT' + "/10_kieks_0_95_95.npy")[:args.n_seed + 2]
                else:
                    raise NotImplementedError

            elif dataset == 'TWH':
                seed_gesture = np.load("../../TWH_dataset/processed/" + 'gesture_TWH' + "/val_2023_v0_014_main-agent.npy")[:args.n_seed + 2]

            seed_gesture = (seed_gesture - data_mean) / data_std
            seed_gesture_vel = seed_gesture[1:] - seed_gesture[:-1]
            seed_gesture_acc = seed_gesture_vel[1:] - seed_gesture_vel[:-1]
            seed_gesture_ = np.concatenate((seed_gesture[2:], seed_gesture_vel[1:], seed_gesture_acc), axis=1)      # (args.n_seed, args.njoints)
            seed_gesture_ = torch.from_numpy(seed_gesture_).float().transpose(0, 1).unsqueeze(0).to(mydevice)
            model_kwargs_['y']['seed'] = seed_gesture_.unsqueeze(2)

        else:
            if args.name == 'DiffuseStyleGesture':
                pad_audio = audio_reshape[-args.n_seed:, i - 1:i]
                model_kwargs_['y']['audio'] = torch.cat((pad_audio, model_kwargs_['y']['audio']), 0).transpose(0, 1)        # attention 3
            elif args.name == 'DiffuseStyleGesture+':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'].transpose(0, 1)  # attention 4
            elif args.name == 'DiffuseStyleGesture++':
                model_kwargs_['y']['audio'] = model_kwargs_['y']['audio'][:-args.n_seed, ...].transpose(0, 1)  # attention 5

            model_kwargs_['y']['seed'] = out_list[-1][..., -args.n_seed:].to(mydevice)

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
        if len(out_list) > 0 and args.n_seed != 0:
            last_poses = out_list[-1][..., -args.n_seed:]        # # (1, model.njoints, 1, args.n_seed)
            out_list[-1] = out_list[-1][..., :-args.n_seed]  # delete last 4 frames
            # if smoothing:
            #     # Extract predictions
            #     last_poses_root_pos = last_poses[:, :12]        # (1, 3, 1, 8)
            #     next_poses_root_pos = sample[:, :12]        # (1, 3, 1, 88)
            #     root_pos = last_poses_root_pos[..., 0]      # (1, 3, 1)
            #     predict_pos = next_poses_root_pos[..., 0]
            #     delta_pos = (predict_pos - root_pos).unsqueeze(-1)      # # (1, 3, 1, 1)
            #     sample[:, :12] = sample[:, :12] - delta_pos
            for j in range(len(last_poses)):
                n = len(last_poses)
                prev = last_poses[..., j]
                next = sample[..., j]
                sample[..., j] = prev * (n - j) / (n + 1) + next * (j + 1) / (n + 1)
        out_list.append(sample)

    if "v0" in args.version:
        motion_feature_division = 3
    elif "v2" in args.version:
        motion_feature_division = 1
    else:
        raise ValueError("wrong version name")

    out_list = [i.detach().data.cpu().numpy()[:, :args.njoints // motion_feature_division] for i in out_list]
    if len(out_list) > 1:
        out_dir_vec_1 = np.vstack(out_list[:-1])
        sampled_seq_1 = out_dir_vec_1.squeeze(2).transpose(0, 2, 1).reshape(batch_size, -1, model.njoints // motion_feature_division)
        out_dir_vec_2 = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
        sampled_seq = np.concatenate((sampled_seq_1, out_dir_vec_2), axis=1)
    else:
        sampled_seq = np.array(out_list[-1]).squeeze(2).transpose(0, 2, 1)
    sampled_seq = sampled_seq[:, args.n_seed:]

    out_poses = np.multiply(sampled_seq[0], data_std) + data_mean
    print(out_poses.shape, real_n_frames)
    out_poses = out_poses[:real_n_frames]
    if dataset == 'BEAT':
        if "v0" in args.version:
            pose2bvh_bugfix(save_dir, prefix, out_poses, pipeline='../process/resource/data_pipe_30fps' + '_speaker' + str(speaker) + '.sav')
        elif "v2" in args.version:
            pose2bvh(save_dir, prefix, out_poses)
        else:
            raise ValueError("wrong version name")
    elif dataset == 'TWH':
        pose2bvh_twh(out_poses, save_dir, prefix, pipeline_path="../process/resource/pipeline_rotmat_62.sav")


def main(args, save_dir, model_path, tst_path=None, max_len=0, skip_timesteps=0, tst_prefix=None, dataset='BEAT', 
         wav_path=None, txt_path=None, wavlm_path=None, word2vector_path=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # sample
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    print(f"Loading checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(mydevice)
    model.eval()
    sample_fn = diffusion.p_sample_loop  # predict x_start

    if tst_path is not None:
        if dataset == 'TWH':
            metadata_path = os.path.join(tst_path, "metadata.csv")
            num_speakers, metadict_byfname, metadict_byindex = load_metadata(metadata_path, "main-agent")
            filenames = sorted(metadict_byfname.keys())
            
        tst_audio_dir = os.path.join(tst_path, 'audio_' + dataset)
        tst_text_dir = os.path.join(tst_path, 'text_' + dataset)

        for i, filename in enumerate(tst_prefix):
            print(f"Processing: {filename}")
            if dataset == 'BEAT':
                speaker_id = speaker_id_dict[int(filename.split('_')[0])]
                speaker = np.zeros([args.style_dim])
                speaker[speaker_id] = 1
            elif dataset == 'TWH':
                _, speaker_id = metadict_byfname[filename]
                speaker = np.zeros([17])
                speaker[speaker_id] = 1
                
            audio_path = os.path.join(tst_audio_dir, filename + '.npy')
            audio = np.load(audio_path)
            text_path = os.path.join(tst_text_dir, filename + '.npy')
            text = np.load(text_path)
            textaudio = np.concatenate((audio, text), axis=-1)
            textaudio = torch.FloatTensor(textaudio)
            textaudio = textaudio.to(mydevice)

            inference(args, save_dir, filename, textaudio, sample_fn, model, n_frames=max_len, smoothing=True, skip_timesteps=skip_timesteps, style=speaker, seed=123456, dataset=dataset)
    else:
        # 20230805 update: generate audiowavlm..., sample from single one
        if dataset == 'TWH':
            from process_TWH_bvh import load_wordvectors, load_audio, load_tsv
        elif dataset == 'BEAT':
            from process_BEAT_bvh import load_wordvectors, load_audio, load_tsv

        wavlm_model, cfg = wavlm_init(wavlm_path, mydevice)
        word2vector = load_wordvectors(fname=word2vector_path)

        wav = load_audio(wav_path, wavlm_model, cfg)
        clip_len = wav.shape[0]
        tsv = load_tsv(txt_path, word2vector, clip_len)
        textaudio = np.concatenate((wav, tsv), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        textaudio = textaudio.to(mydevice)
        speaker = np.zeros([17])
        speaker[0] = 1      # random choice will be great
        filename = 'tts'
        inference(args, save_dir, filename, textaudio, sample_fn, model, n_frames=max_len, smoothing=True,
                  skip_timesteps=skip_timesteps, style=speaker, seed=123456, dataset=dataset)


if __name__ == '__main__':
    '''
    python sample.py --config=./configs/DiffuseStyleGesture.yml --gpu 7 --model_path "./BEAT_mymodel4_512_v0/model001260000.pt" --max_len 0 --tst_prefix '2_scott_0_1_1'
    '''
    parser = argparse.ArgumentParser(description='DiffuseStyleGesture')
    parser.add_argument('--config', default='./configs/DiffuseStyleGesture.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--tst_prefix', nargs='+')
    parser.add_argument('--no_cuda', type=list, default=['0'])
    parser.add_argument('--model_path', type=str, default='./model000450000.pt')
    parser.add_argument('--tst_path', type=str, default=None)
    parser.add_argument('--wav_path', type=str, default=None)
    parser.add_argument('--txt_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='sample_dir')
    parser.add_argument('--max_len', type=int, default=0)
    parser.add_argument('--skip_timesteps', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='BEAT')
    parser.add_argument('--wavlm_path', type=str, default='./WavLM/WavLM-Large.pt')
    parser.add_argument('--word2vector_path', type=str, default='./crawl-300d-2M.vec')
    
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    # pprint(config)
    config = EasyDict(config)

    assert config.name in ['DiffuseStyleGesture', 'DiffuseStyleGesture+', 'DiffuseStyleGesture++']
    if config.name == 'DiffuseStyleGesture+':
        config.cond_mode = 'cross_local_attention4_style1_sample'
    elif config.name == 'DiffuseStyleGesture':
        config.cond_mode = 'cross_local_attention3_style1_sample'
    elif config.name == 'DiffuseStyleGesture++':
        config.cond_mode = 'cross_local_attention5_style1_sample'
        
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

    device_name = 'cuda:' + args.gpu
    mydevice = torch.device('cuda:' + config.gpu)
    torch.cuda.set_device(int(config.gpu))
    args.no_cuda = args.gpu

    batch_size = 1

    model_root = config.model_path.split('/')[1]
    model_spicific = config.model_path.split('/')[-1].split('.')[0]
    config.save_dir = "./" + model_root + '/' + 'sample_dir_' + model_spicific + '/'
    if config.tst_prefix is not None:
        config.tst_path = "../../" + config.dataset + "_dataset/processed/"

    print('model_root', model_root, 'tst_path', config.tst_path, 'save_dir', config.save_dir)

    main(config, config.save_dir, config.model_path, tst_path=config.tst_path, max_len=config.max_len,
         skip_timesteps=config.skip_timesteps, tst_prefix=config.tst_prefix, dataset=config.dataset, 
         wav_path=config.wav_path, txt_path=config.txt_path, wavlm_path=config.wavlm_path, word2vector_path=config.word2vector_path)

