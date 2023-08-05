from pymo_TWH.parsers import BVHParser
from pymo_TWH.preprocessing import *
from pymo_TWH.viz_tools import *
from pymo_TWH.writers import *
from tool import *
import os
import numpy as np
import pandas as pd
import pdb
from sklearn.pipeline import Pipeline
import joblib as jl
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
import torch
import torch.nn.functional as F
import librosa
import string
import io
from tqdm import tqdm
import h5py
import argparse

bone_names = ['body_world', 'b_root', 'b_l_upleg', 'b_l_leg', 'b_l_foot_twist', 'b_l_foot', 'b_r_upleg', 'b_r_leg', 'b_r_foot_twist', 'b_r_foot', 'b_spine0', 'b_spine1', 'b_spine2', 'b_spine3', 'b_neck0', 'b_head', 'b_l_shoulder', 'p_l_scap', 'b_l_arm', 'b_l_arm_twist', 'b_l_forearm', 'b_l_wrist_twist', 'b_l_wrist', 'b_l_thumb0', 'b_l_thumb1', 'b_l_thumb2', 'b_l_thumb3', 'b_l_index1', 'b_l_index2', 'b_l_index3', 'b_l_middle1', 'b_l_middle2', 'b_l_middle3', 'b_l_ring1', 'b_l_ring2', 'b_l_ring3', 'b_l_pinky1', 'b_l_pinky2', 'b_l_pinky3', 'b_r_shoulder', 'p_r_scap', 'b_r_arm', 'b_r_arm_twist', 'b_r_forearm', 'b_r_wrist_twist', 'b_r_wrist', 'b_r_index1', 'b_r_index2', 'b_r_index3', 'b_r_ring1', 'b_r_ring2', 'b_r_ring3', 'b_r_middle1', 'b_r_middle2', 'b_r_middle3', 'b_r_pinky1', 'b_r_pinky2', 'b_r_pinky3', 'b_r_thumb0', 'b_r_thumb1', 'b_r_thumb2', 'b_r_thumb3']


def load_bvh(bvhfile, dump_pipeline=False, mode='expmap'):
    assert mode in ('expmap', 'euler', 'quat', 'rotmat'), "mode must be one of 'expmap', 'euler', 'quat', 'rotmat'"
    parser = BVHParser()
    parsed_data = parser.parse(bvhfile)

    if mode == 'expmap':
        mexp_full = Pipeline([
            ('jtsel', JointSelector(bone_names, include_root=True)),
            ('param', MocapParameterizer(mode)),
            ('cnst', ConstantsRemover_withroot()),
            ('np', Numpyfier()),
        ])

    elif mode == 'euler' or mode == 'quat' or mode == 'rotmat':
        mexp_full = Pipeline([
            ('jtsel', JointSelector(bone_names, include_root=False)),
            ('np', Numpyfier()),
        ])
    if mode is not 'rotmat':
        out_data = mexp_full.fit_transform([parsed_data])[0]
    else:
        out_data = mexp_full.fit_transform([parsed_data])
    if dump_pipeline:
        jl.dump(mexp_full, "pipeline_" + mode + '_' + str(len(bone_names)) + ".sav")

    if mode == 'rotmat':
        # euler -> rotation matrix
        out_data = out_data.reshape((out_data.shape[0], out_data.shape[1], -1, 6))  # 3 pos (XYZ), 3 rot (ZXY)
        out_matrix = np.zeros(
            (out_data.shape[0], out_data.shape[1], out_data.shape[2], 12))  # 3 pos, 1 rot matrix (9 elements)
        for i in range(out_data.shape[0]):  # mirror
            for j in range(out_data.shape[1]):  # frames
                for k in range(out_data.shape[2]):  # joints
                    out_matrix[i, j, k, :3] = out_data[i, j, k, :3]  # positions
                    r = R.from_euler('ZXY', out_data[i, j, k, 3:], degrees=True)
                    out_matrix[i, j, k, 3:] = r.as_matrix().flatten()  # rotations
        out_matrix = out_matrix.reshape((out_data.shape[0], out_data.shape[1], -1))
        return out_matrix[0]
    else:
        return out_data


def wavlm_init(wavlm_model_path, device=torch.device('cuda:0')):
    import sys
    [sys.path.append(i) for i in ['./WavLM', '../process/WavLM']]
    from WavLM import WavLM, WavLMConfig
    checkpoint = torch.load(wavlm_model_path, map_location=torch.device('cpu'))     # load the pre-trained checkpoints
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
            rep.append(model.extract_features(wav_input_16khz[i:i+10])[0])
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
    mfcc_f = calculate_mfcc(wav, sr)        # (7205, 40)
    melspec_f = calculate_spectrogram(wav, sr)      # (7205, 64)
    prosody = extract_prosodic_features(audiofile)      # (7199, 4)
    crop_length = min(mfcc_f.shape[0], melspec_f.shape[0], prosody.shape[0])
    wavlm_f = wav2wavlm(wavlm_model, wav_input_16khz, cfg, device)      # [12201, 1024]
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
        onsets_resample[i-1] = (max(silence[int(xp[i-1]):int(xp[i])])) == 1
    audio_f = np.concatenate((mfcc_f[:crop_length], melspec_f[:crop_length], prosody[:crop_length], wavlm_f, onsets_resample.reshape(-1, 1)), axis=1)
    return audio_f


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
    try:
        clip_length = int(sentence[-1][1] * 30)
    except:
        clip_length = 0
    return sentence, clip_length


def load_wordvectors(fname):        # take about 03:27
    print("Loading word2vector ...")
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(v) for v in tokens[1:]])
    return data


def load_tsv(tsvpath, word2vector, clip_len):
    # Align txt with audio

    sentence, _ = load_tsv_unclipped(tsvpath)
    textfeatures = np.zeros([clip_len, 300 + 2])
    textfeatures[:, -1] = 1

    for wi, (start, end, raw_word) in enumerate(sentence):
        has_laughter = "#" in raw_word
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
        textfeatures[start_frame:end_frame, -2] = has_laughter
    return textfeatures


def pose2bvh(predicted_gesture, output_dir, name, pipeline_path="./pipeline_expmap_25.sav"):
    mode = pipeline_path.split("_")[1]
    pipeline = jl.load(pipeline_path)

    # smoothing
    n_poses = predicted_gesture.shape[0]
    out_poses = np.zeros((n_poses, predicted_gesture.shape[1]))
    for i in range(predicted_gesture.shape[1]):
        out_poses[:, i] = savgol_filter(predicted_gesture[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal

    if mode == 'rotmat':
        # rotation matrix to euler angles
        out_poses = out_poses.reshape((out_poses.shape[0], -1, 12))  # (n_frames, n_joints, 12)
        out_data = np.zeros((out_poses.shape[0], out_poses.shape[1], 6))
        for i in range(out_poses.shape[0]):  # frames
            for j in range(out_poses.shape[1]):  # joints
                out_data[i, j, :3] = out_poses[i, j, :3]
                r = R.from_matrix(out_poses[i, j, 3:].reshape(3, 3))
                out_data[i, j, 3:] = r.as_euler('ZXY', degrees=True).flatten()

        out_data = out_data.reshape(out_data.shape[0], -1)
        bvh_data = pipeline.inverse_transform([out_data])[0]
    else:
        bvh_data = pipeline.inverse_transform([predicted_gesture])[0]
    writer = BVHWriter()
    with open(os.path.join(output_dir, f"{name}.bvh"), 'w') as f:
        writer.write(bvh_data, f, framerate=30)


def load_metadata(metadata, participant):
    assert participant in ("main-agent", "interloctr"), "`participant` must be either 'main-agent' or 'interloctr'"

    metadict_byfname = {}
    metadict_byindex = {}
    speaker_ids = []
    finger_info = []
    with open(metadata, "r") as f:
        # NOTE: The first line contains the csv header so we skip it
        for i, line in enumerate(f.readlines()[1:]):
            (
                fname,
                main_speaker_id,
                main_has_finger,
                ilocutor_speaker_id,
                ilocutor_has_finger,
            ) = line.strip().split(",")

            if participant == "main-agent":
                has_finger = (main_has_finger == "finger_incl")
                speaker_id = int(main_speaker_id) - 1
            else:
                has_finger = (ilocutor_has_finger == "finger_incl")
                speaker_id = int(ilocutor_speaker_id) - 1

            finger_info.append(has_finger)
            speaker_ids.append(speaker_id)

            metadict_byindex[i] = has_finger, speaker_id
            metadict_byfname[fname + f"_{participant}"] = has_finger, speaker_id

    speaker_ids = np.array(speaker_ids)
    finger_info = np.array(finger_info)
    num_speakers = np.unique(speaker_ids).shape[0]
    # assert num_speakers == spks.max(), "Error speaker info!"
    # print("Number of speakers: ", num_speakers)
    # print("Has Finger Ratio:", np.mean(finger_info))

    return num_speakers, metadict_byfname, metadict_byindex


def prepare_data(data_path, dataset_type, participant, mode, save_path, wavlm_model, word2vector, preload, version, 
                 debug=False, device=torch.device('cuda:0')):
    assert dataset_type in ("trn", "val", "tst"), "`dataset_type` must be either 'trn', 'val', or 'tst'"
    assert participant in ("main-agent", "interloctr"), "`participant` must be either 'main-agent' or 'interloctr'"
    motion_save_path = os.path.join(save_path, dataset_type, participant, 'gesture_TWH')
    audio_save_path = os.path.join(save_path, dataset_type, participant, 'audio_TWH')
    text_save_path = os.path.join(save_path, dataset_type, participant, 'text_TWH')
    if not os.path.exists(motion_save_path):
        os.makedirs(motion_save_path)
    if not os.path.exists(audio_save_path):
        os.makedirs(audio_save_path)
    if not os.path.exists(text_save_path):
        os.makedirs(text_save_path)

    dataroot = os.path.join(data_path, dataset_type)
    metadata_path = os.path.join(dataroot, "metadata.csv")
    num_speakers, metadict_byfname, metadict_byindex = load_metadata(metadata_path, participant)
    filenames = sorted(metadict_byfname.keys())

    wavdir = os.path.join(dataroot, participant, "wav")
    tsvdir = os.path.join(dataroot, participant, "tsv")
    bvhdir = os.path.join(dataroot, participant, "bvh")
    
    if debug:
        all_filenames = ['trn_2023_v0_169_main-agent']
    else:
        all_filenames = filenames
    
    # with h5py.File(f"{dataset_type}_{participant}_v0.h5", "w") as h5:
    with h5py.File(f"TWH_" + version + ".h5", "w") as h5:
        for i, filename in enumerate(all_filenames):
            print(f"Processing {i+1}/{len(filenames)}: {filename}", end="\r")
            g_data = h5.create_group(str(i))
            hasfinger, speaker_id = metadict_byfname[filename]
            wavpath = os.path.join(wavdir, filename + ".wav")
            tsvpath = os.path.join(tsvdir, filename + ".tsv")
            bvhpath = os.path.join(bvhdir, filename + ".bvh")

            if not preload:
                if dataset_type == 'trn' or dataset_type == 'val':
                    # process gesture
                    dump_pipeline = (filename == 'trn_2023_v0_002_main-agent')
                    bvh = load_bvh(bvhpath, dump_pipeline=dump_pipeline, mode=mode)
                    np.save(os.path.join(motion_save_path, filename + ".npy"), bvh)

                # process audio
                if os.path.exists(os.path.join(audio_save_path, filename + ".npy")):
                    print(f'{filename} exist')
                    continue
                wav = load_audio(wavpath, wavlm_model, cfg, device=device)
                if dataset_type == 'trn' or dataset_type == 'val':
                    np.save(os.path.join(audio_save_path, filename + ".npy"), wav)
                else:
                    np.save(os.path.join(audio_save_path, filename + '_' + str(speaker_id) + ".npy"), wav)

                # process text
                if os.path.exists(os.path.join(text_save_path, filename + ".npy")):
                    print(f'{filename} exist')
                    continue
                # clip_len = np.load(os.path.join(audio_save_path, filename + ".npy")).shape[0]
                clip_len = wav.shape[0]
                tsv = load_tsv(tsvpath, word2vector, clip_len)
                np.save(os.path.join(text_save_path, filename + ".npy"), tsv)
            else:
                if dataset_type == 'trn' or dataset_type == 'val':
                    # process gesture
                    bvh = np.load(os.path.join(motion_save_path, filename + ".npy"))

                # process audio
                wav = np.load(os.path.join(audio_save_path, filename + ".npy"))

                # process text
                tsv = np.load(os.path.join(text_save_path, filename + ".npy"))

            if dataset_type == 'trn' or dataset_type == 'val':
                clip_len = min(bvh.shape[0], wav.shape[0], tsv.shape[0])
                bvh = bvh[:clip_len]
                wav = wav[:clip_len]
                tsv = tsv[:clip_len]

                g_data.create_dataset("has_finger", data=[hasfinger])
                g_data.create_dataset("speaker_id", data=[speaker_id])
                g_data.create_dataset("gesture", data=bvh, dtype=np.float32)
                g_data.create_dataset("audio", data=wav, dtype=np.float32)
                g_data.create_dataset("text", data=tsv, dtype=np.float32)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='process_twh_bvh.py')
    parser.add_argument('--dataroot', type=str, default='/ceph/datasets/Genea2023/genea2023_dataset/')
    parser.add_argument('--mode', type=str, default='rotmat')
    parser.add_argument('--version', type=str, default='v0')
    parser.add_argument('--wavlm_path', type=str, default='./WavLM/WavLM-Large.pt')
    parser.add_argument('--word2vector_path', type=str, default='./crawl-300d-2M.vec')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument("--step", type=str, default='1')
    parser.add_argument('--save_path', type=str, default='./dataset/processed/')
    parser.add_argument('--debug', type=str, default='False')

    args = parser.parse_args()
    
    debug = args.debug == 'True'
    
    if args.step == '1':
        wavlm_path = args.wavlm_path
        word2vector_path = args.word2vector_path
        wavlm_model, cfg = wavlm_init(wavlm_path, torch.device('cuda:' + args.gpu))
        word2vector = load_wordvectors(fname=word2vector_path)
        for dataset_type in ['trn']:     # 'trn', 'val', 'tst'
            for participant in ['main-agent']:      # , 'interloctr'
                prepare_data(args.dataroot, dataset_type, participant, args.mode, args.save_path, wavlm_model, 
                             word2vector, preload=False, version=args.version, debug=debug, device=torch.device('cuda:' + args.gpu))
    elif args.step == '2':
        wavlm_model, cfg = None, None
        word2vector = None
        for dataset_type in ['trn']:     # 'trn', 'val', 'tst'
            for participant in ['main-agent']:      # , 'interloctr'
                prepare_data(args.dataroot, dataset_type, participant, args.mode, args.save_path, wavlm_model, 
                             word2vector, preload=True, version=args.version, debug=debug)



