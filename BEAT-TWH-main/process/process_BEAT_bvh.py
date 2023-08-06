import argparse
import os
import glob
from pathlib import Path
import torch
import librosa
import numpy as np
from sklearn.pipeline import Pipeline
from tool import *
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from pymo.viz_tools import *
from pymo.writers import *
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import joblib as jl
import textgrid
from utils.data_utils import SubtitleWrapper, normalize_string
import csv
import io
from tqdm import tqdm
import string
import h5py

from anim import bvh, quat, txform
from beat_data_proc.MyBVH import load_bvh_data
from scipy.spatial.transform import Rotation

import pdb

target_joints = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'HeadEnd',
                 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                 'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4',
                 'RightHandRing', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4',
                 'RightHandPinky', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4',
                 'RightHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4',
                 'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4',
                 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                 'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4',
                 'LeftHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4',
                 'LeftHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4',
                 'LeftHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4',
                 'LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb4',
                 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightForeFoot', 'RightToeBase', 'RightToeBaseEnd',
                 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftForeFoot', 'LeftToeBase', 'LeftToeBaseEnd']

order = 'XYZ'       # 'XYZ', 'ZXY'


# print(len(target_joints))       # 74

def process_bvh_bugfix(gesture_filename):  # TODO: 20230723 not working yet
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=30, keep_all=False)),
        # ('root', RootTransformer('hip_centric')),
        # ('mir', Mirror(axis='X', append=True)),
        ('jtsel', JointSelector(target_joints, include_root=True)),  # 'Hips'
        # ('cnst', ConstantsRemover()),       # (2, 6061, n)  -> (2, 6061, n+6)
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    if not os.path.exists('./resource'):
        os.makedirs('./resource')
    speaker = gesture_filename.split('/')[-1].split('_')[0]
    jl.dump(data_pipe, os.path.join('./resource', 'data_pipe_30fps_speaker' + speaker + '.sav'))

    print(out_data.shape)

    # euler -> rotation matrix
    out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 3))
    out_matrix = np.zeros((out_data.shape[0], out_data.shape[1], out_data.shape[2], 9))
    for i in range(out_data.shape[0]):  # mirror
        for j in range(out_data.shape[1]):  # frames
            r = R.from_euler(order, out_data[i, j], degrees=True)
            out_matrix[i, j] = r.as_matrix().reshape(out_data.shape[2], 9)
    out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))

    return out_matrix[0]


def preprocess_animation(animation_file, fps=120):
    # TODO
    pass


def euler2mat(angles, euler_orders):
    # TODO
    pass


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    # todo
    pass


def process_bvh(gesture_filename):
    all_poses, parents, dt, order, njoints = preprocess_animation(gesture_filename, fps=30)
    return all_poses


def pose2bvh_bugfix(save_path, filename_prefix, poses, pipeline='./resource/data_pipe_30fps.sav'):
    writer = BVHWriter()
    pipeline = jl.load(pipeline)

    # smoothing
    n_poses = poses.shape[0]
    out_poses = np.zeros((n_poses, poses.shape[1]))

    for i in range(poses.shape[1]):
        out_poses[:, i] = savgol_filter(poses[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal

    # rotation matrix to euler angles
    out_poses = out_poses.reshape((out_poses.shape[0], -1, 9))
    out_poses = out_poses.reshape((out_poses.shape[0], out_poses.shape[1], 3, 3))
    out_euler = np.zeros((out_poses.shape[0], out_poses.shape[1] * 3))
    for i in range(out_poses.shape[0]):  # frames
        r = R.from_matrix(out_poses[i])
        out_euler[i] = r.as_euler(order, degrees=True).flatten()

    bvh_data = pipeline.inverse_transform([out_euler])

    out_bvh_path = os.path.join(save_path, filename_prefix + '_generated.bvh')
    with open(out_bvh_path, 'w') as f:
        writer.write(bvh_data[0], f)


def write_bvh(filename, V_root_pos, V_root_rot, V_lpos, V_lrot, parents, names, order, dt, start_position=None, start_rotation=None):
    # TODO
    pass


def pose2bvh(save_path, filename_prefix, poses, smoothing=True):
    # TODO
    pass


def wavlm_init(wavlm_model_path, device=torch.device('cuda:0')):
    import sys
    [sys.path.append(i) for i in ['./WavLM', '../process/WavLM']]
    from WavLM import WavLM, WavLMConfig
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))  # load the pre-trained checkpoints
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, cfg


def wav2wavlm(model, wav_input_16khz, cfg, device=torch.device('cuda:0')):
    with torch.no_grad():
        wav_input_16khz = wav_input_16khz.to(device)
        if cfg.normalize:
            wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz, wav_input_16khz.shape)
        wav_len = wav_input_16khz.shape[0]
        chunk_len = 16000 * 5
        num_chunks = wav_len // chunk_len + 1
        wav_input_16khz = torch.nn.functional.pad(wav_input_16khz, (0, chunk_len * num_chunks - wav_len))
        wav_input_16khz = wav_input_16khz.reshape(num_chunks, chunk_len)
        rep = []
        for i in range(0, num_chunks, 10):
            rep.append(model.extract_features(wav_input_16khz[i:i + 10])[0])
        rep = torch.cat(rep, dim=0)
        del wav_input_16khz
        rep = rep.reshape(-1, rep.shape[-1]).detach().cpu()
        return rep


def load_audio(audiofile, wavlm_model, cfg, device=torch.device('cuda:0')):
    wav, sr = librosa.load(audiofile, sr=16000)
    wav_input_16khz = torch.from_numpy(wav).to(torch.float32)
    '''
    kernel_size=(10,), stride=(5,)
    kernel_size=(3,), stride=(2,)
    kernel_size=(3,), stride=(2,)
    kernel_size=(3,), stride=(2,)
    kernel_size=(3,), stride=(2,)
    kernel_size=(2,), stride=(2,)
    kernel_size=(2,), stride=(2,)
    [Lin+2×padding−dilation×(kernel_size−1)−1]/stride + 1
    (((((((x -10)/5 + 1 - 3) / 2 + 1 - 3) / 2 + 1 - 3) / 2 + 1 - 3) / 2 + 1 - 2) / 2 + 1 - 2) / 2 + 1  -> (x-80)/320
    '''
    # wav_input_16khz = torch.randn(1, 10000)     # (1, 10000) -> (1, 512, 1999) -> (1, 512, 999) -> (1, 512, 499) -> (1, 512, 249) -> (1, 512, 124), -> (1, 512, 62) -> (1, 512, 31)
    mfcc_f = calculate_mfcc(wav, sr)  # (7205, 40)
    melspec_f = calculate_spectrogram(wav, sr)  # (7205, 64)
    prosody = extract_prosodic_features(audiofile)  # (7199, 4)
    crop_length = min(mfcc_f.shape[0], melspec_f.shape[0], prosody.shape[0])
    wavlm_f = wav2wavlm(wavlm_model, wav_input_16khz, cfg, device)  # [12201, 1024]
    wavlm_f = F.interpolate(wavlm_f.unsqueeze(0).transpose(1, 2), size=crop_length, align_corners=True,
                            mode='linear').transpose(1, 2).squeeze(0)
    onsets_f, _ = extract_onsets(audiofile)
    # x = np.linspace(0, len(wav) - 1, num=len(wav))
    xp = np.linspace(0, len(wav) - 1, num=crop_length + 1)
    # audio_hfc = np.interp(xp, x, y)     # np.count_nonzero(audio_hfc)
    silence = np.array([0.] * len(wav))
    silence[(np.clip(onsets_f * 16000, 0, len(wav) - 1)).astype('int64')] = 1
    onsets_resample = np.array([0.] * crop_length)
    for i in range(1, crop_length + 1):
        onsets_resample[i - 1] = (max(silence[int(xp[i - 1]):int(xp[i])])) == 1
    audio_f = np.concatenate(
        (mfcc_f[:crop_length], melspec_f[:crop_length], prosody[:crop_length], wavlm_f, onsets_resample.reshape(-1, 1)),
        axis=1)
    return audio_f


def Grid2tsv(TextGrid_path):
    tg = textgrid.TextGrid()
    tg.read(TextGrid_path)  # 是文件名
    with open(TextGrid_path.replace('.TextGrid', '.tsv'), 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        for key in tg.tiers[0]:
            if key.mark == '': continue
            tsv_w.writerow([key.minTime, key.maxTime, key.mark])


def load_wordvectors(fname):  # take about 03:27
    print("Loading word2vector ...")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(v) for v in tokens[1:]])
    return data


def load_tsv_unclipped(tsvfile):
    sentence = []
    with open(tsvfile, "r") as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split("\t")
            if len(line) == 3:
                start, end, raw_word = line
                start = float(start)
                end = float(end)
                sentence.append([start, end, raw_word])

    return sentence


def load_tsv(tsvpath, word2vector, clip_len):
    # Align txt with audio

    sentence = load_tsv_unclipped(tsvpath)
    textfeatures = np.zeros([clip_len, 300 + 1])
    textfeatures[:, -1] = 1

    for wi, (start, end, raw_word) in enumerate(sentence):
        start_frame = int(start * 30)
        end_frame = int(end * 30)
        textfeatures[start_frame:end_frame, -1] = 0

        word = raw_word.translate(str.maketrans('', '', string.punctuation))
        word = word.strip()
        word = word.replace("  ", " ")

        if len(word) > 0:
            if word[0] == " ":
                word = word[1:]

        if " " in word:
            ww = word.split(" ")
            subword_duration = (end_frame - start_frame) / len(ww)
            for j, w in enumerate(ww):
                vector = word2vector.get(w)
                if vector is not None:
                    ss = start_frame + int(subword_duration * j)
                    ee = start_frame + int(subword_duration * (j + 1))
                    textfeatures[ss:ee, :300] = vector
        else:
            vector = word2vector.get(word)
            if vector is not None:
                textfeatures[start_frame:end_frame, :300] = vector
    return textfeatures


def pre_processing(base_path):
    """
    Handle the problem of inconsistency between real frames and actual frames.
    There are a lot of bvh files where the total number of frames doesn't match the actual motion data, for example:
    2_scott_0_103_110.bvh  (47934+430 frames against 48285 frames)
    """
    p = BVHParser()
    for speaker in os.listdir(base_path):
        print('Processing speaker {}'.format(speaker))
        gesture_path = os.path.join(base_path, speaker)
        bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))
        for v_i, bvh_file in enumerate(bvh_files):
            name = os.path.split(bvh_file)[1][:-4]
            print('check:', name)
            try:
                p.parse(bvh_file)
                bvh.load(bvh_file)
            except:
                print('process: ' + name)
                bvh_file_ = open(bvh_file, 'r')
                content = bvh_file_.readlines()
                length = len(content)  # 15_carlos_0_8_8 -> 5899
                correct_frames = length - 431
                content[429] = 'Frames: ' + str(correct_frames) + '\n'
                file = open(bvh_file, 'w')
                file.writelines(content)
                bvh_file_.close()
                file.close()


def process_T_pose(base_path):
    for speaker in os.listdir(base_path):
        if speaker != '2':
            continue
        print('Processing speaker {}'.format(speaker))
        gesture_path = os.path.join(base_path, speaker)
        bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))
        for v_i, bvh_file in enumerate(bvh_files):
            name = os.path.split(bvh_file)[1][:-4]
            print('process:', name)
            bvh_file_ = open(bvh_file, 'r')
            content = bvh_file_.readlines()
            bvh_file_.close()
            for i, line in enumerate(content):
                if 'OFFSET' in line:
                    line = line.strip('\n').split(' ')
                    x = float(line[-3])
                    y = float(line[-2])
                    z = float(line[-1])
                    line[-3] = str(0.0 - x)
                    line[-1] = str(0.0 - z)
                    line = ' '.join(line) + '\n'
                    content[i] = line
                if i >= 431:
                    line = line.strip().replace('  ', ' ').split(' ')
                    line[4] = str(float(line[4]) - 180.0)
                    line[5] = str(0.0 - float(line[5]))
                    for j in range(2 + 6, len(line), 3):
                        line[j] = str(0.0 - float(line[j]))
                        line[j-2] = str(0.0 - float(line[j-2]))
                    line = ' '.join(line) + '\n'
                    content[i] = line

            # with open(bvh_file.replace('2_scott_0_1_1.bvh', '2_scott_0_1_1_re.bvh'), 'w') as output:
            #     output.writelines(content)

            file = open(bvh_file, 'w')
            file.writelines(content)
            file.close()
            

def make_gesture_dataset(base_path, save_path, preload=False, wavlm_model=None, cfg=None, word2vector=None,
                         device=torch.device('cuda:0'), version='v0'):
    motion_save_path = os.path.join(save_path, 'gesture_BEAT')
    audio_save_path = os.path.join(save_path, 'audio_BEAT')
    text_save_path = os.path.join(save_path, 'text_BEAT')
    if not os.path.exists(motion_save_path):
        os.makedirs(motion_save_path)
    if not os.path.exists(audio_save_path):
        os.makedirs(audio_save_path)
    if not os.path.exists(text_save_path):
        os.makedirs(text_save_path)

    if not preload:
        for speaker in os.listdir(base_path):
            if speaker not in ['2', '10']:
                continue
            print('Processing speaker {}'.format(speaker))
            gesture_path = os.path.join(base_path, speaker)
            bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))
            for v_i, bvh_file in enumerate(bvh_files):
                name = os.path.split(bvh_file)[1][:-4]
                print(f"Processing {v_i + 1}/{len(bvh_files)}: {name}")

                # process gesture
                if os.path.exists(os.path.join(motion_save_path, name + ".npy")):
                    print(f'gesture {name} exist')
                else:
                    if 'v0' in version:
                        poses = process_bvh_bugfix(bvh_file)
                    elif ('v1' in version) or ('v2' in version):
                        # TODO 
                        pass
                    else:
                        raise NotImplementedError
                    np.save(os.path.join(motion_save_path, name + ".npy"), poses)

                # process audio
                if os.path.exists(os.path.join(audio_save_path, name + ".npy")):
                    print(f'audio {name} exist')
                else:
                    wavpath = bvh_file[:-4] + '.wav'
                    wav = load_audio(wavpath, wavlm_model, cfg, device)
                    np.save(os.path.join(audio_save_path, name + ".npy"), wav)

                # process text
                if os.path.exists(os.path.join(text_save_path, name + ".npy")):
                    print(f'text {name} exist')
                else:
                    try:
                        clip_len = wav.shape[0]
                    except:
                        wav = np.load(os.path.join(audio_save_path, name + ".npy"))
                        print(f'load wav from {audio_save_path}', wav.shape)
                    tsvpath = bvh_file[:-4] + '.TextGrid'
                    Grid2tsv(tsvpath)
                    tsv = load_tsv(tsvpath.replace('.TextGrid', '.tsv'), word2vector, clip_len)
                    np.save(os.path.join(text_save_path, name + ".npy"), tsv)

    if preload:
        with h5py.File(f"BEAT_" + version + ".h5", "w") as h5:
            total_index = 0
            for speaker in os.listdir(base_path):
                if speaker not in ['2', '10']:
                    continue
                print('Processing speaker {}'.format(speaker))
                gesture_path = os.path.join(base_path, speaker)
                bvh_files = sorted(glob.glob(gesture_path + "/*.bvh"))
                for v_i, bvh_file in enumerate(bvh_files):
                    name = os.path.split(bvh_file)[1][:-4]
                    if name == '2_scott_0_1_1': continue  # pick a small files to test
                    print(f"Processing {v_i + 1}/{len(bvh_files)}: {name}")
                    g_data = h5.create_group(str(total_index))

                    poses = np.load(os.path.join(motion_save_path, name + ".npy"))
                    wav = np.load(os.path.join(audio_save_path, name + ".npy"))
                    tsv = np.load(os.path.join(text_save_path, name + ".npy"))
                    clip_len = min(poses.shape[0], wav.shape[0], tsv.shape[0])
                    poses = poses[:clip_len]
                    wav = wav[:clip_len]
                    tsv = tsv[:clip_len]
                    g_data.create_dataset("speaker_id", data=[speaker])  # TODO style
                    g_data.create_dataset("gesture", data=poses, dtype=np.float32)
                    g_data.create_dataset("audio", data=wav, dtype=np.float32)
                    g_data.create_dataset("text", data=tsv, dtype=np.float32)

                    total_index += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("db_path", type=Path)
    parser.add_argument("save_path", type=Path)
    parser.add_argument("wavlm_model_path", type=Path)
    parser.add_argument("word2vec_model_path", type=Path)
    parser.add_argument("version", type=str, default="v0")
    parser.add_argument("step", type=str)
    parser.add_argument("device", type=str, default="cuda:0")
    args = parser.parse_args()

    wavlm_model_path = args.wavlm_model_path
    word2vec_model_path = args.word2vec_model_path
    
    assert args.step in ["step1", "step2", "step3", "step4"]
    if args.step == "step1":
        pre_processing(args.db_path)
    elif args.step == "step2":
        process_T_pose(args.db_path)
    elif args.step == "step3":
        preload = False

        # device = None
        # wavlm_model, cfg = None, None
        # word2vector = None
        device = torch.device(args.device)
        wavlm_model, cfg = wavlm_init(wavlm_model_path, device)
        word2vector = load_wordvectors(fname=word2vec_model_path)

        make_gesture_dataset(args.db_path, args.save_path, preload, wavlm_model, cfg, word2vector, device, args.version)
    elif args.step == "step4":
        preload = True
        device = None
        wavlm_model, cfg = None, None
        word2vector = None
        make_gesture_dataset(args.db_path, args.save_path, preload, wavlm_model, cfg, word2vector, device, args.version)


if __name__ == '__main__':
    main()
