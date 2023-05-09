import json
import pdb
import numpy as np
from omegaconf import DictConfig
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
[sys.path.append(i) for i in ['.', '..', '../../ubisoft-laforge-ZeroEGGS-main/ZEGGS']]

from anim import bvh, quat, txform
from utils_zeggs import write_bvh
import torch
from scipy.signal import savgol_filter


bone_names = [
        "Hips",
        "Spine",
        "Spine1",
        "Spine2",
        "Spine3",
        "Neck",
        "Neck1",
        "Head",
        "HeadEnd",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "RightHandThumb1",
        "RightHandThumb2",
        "RightHandThumb3",
        "RightHandThumb4",
        "RightHandIndex1",
        "RightHandIndex2",
        "RightHandIndex3",
        "RightHandIndex4",
        "RightHandMiddle1",
        "RightHandMiddle2",
        "RightHandMiddle3",
        "RightHandMiddle4",
        "RightHandRing1",
        "RightHandRing2",
        "RightHandRing3",
        "RightHandRing4",
        "RightHandPinky1",
        "RightHandPinky2",
        "RightHandPinky3",
        "RightHandPinky4",
        "RightForeArmEnd",
        "RightArmEnd",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "LeftHandThumb1",
        "LeftHandThumb2",
        "LeftHandThumb3",
        "LeftHandThumb4",
        "LeftHandIndex1",
        "LeftHandIndex2",
        "LeftHandIndex3",
        "LeftHandIndex4",
        "LeftHandMiddle1",
        "LeftHandMiddle2",
        "LeftHandMiddle3",
        "LeftHandMiddle4",
        "LeftHandRing1",
        "LeftHandRing2",
        "LeftHandRing3",
        "LeftHandRing4",
        "LeftHandPinky1",
        "LeftHandPinky2",
        "LeftHandPinky3",
        "LeftHandPinky4",
        "LeftForeArmEnd",
        "LeftArmEnd",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToeBase",
        "RightToeBaseEnd",
        "RightLegEnd",
        "RightUpLegEnd",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToeBase",
        "LeftToeBaseEnd",
        "LeftLegEnd",
        "LeftUpLegEnd"
    ]


def preprocess_animation(animation_file, fps=60):
    anim_data = bvh.load(animation_file)       #  'rotations' (8116, 75, 3), 'positions', 'offsets' (75, 3), 'parents', 'names' (75,), 'order' 'zyx', 'frametime' 0.016667
    nframes = len(anim_data["rotations"])

    if fps != 60 :
        rate = 60 // fps
        anim_data["rotations"] = anim_data["rotations"][0:nframes:rate]
        anim_data["positions"] = anim_data["positions"][0:nframes:rate]
        dt = 1 / fps
        nframes = anim_data["positions"].shape[0]
    else:
        dt = anim_data["frametime"]

    njoints = len(anim_data["parents"])

    lrot = quat.unroll(quat.from_euler(np.radians(anim_data["rotations"]), anim_data["order"]))
    lpos = anim_data["positions"]
    grot, gpos = quat.fk(lrot, lpos, anim_data["parents"])
    # Find root (Projected hips on the ground)
    root_pos = gpos[:, anim_data["names"].index("Spine2")] * np.array([1, 0, 1])
    # Root direction
    root_fwd = quat.mul_vec(grot[:, anim_data["names"].index("Hips")], np.array([[0, 0, 1]]))
    root_fwd[:, 1] = 0
    root_fwd = root_fwd / np.sqrt(np.sum(root_fwd * root_fwd, axis=-1))[..., np.newaxis]
    # Root rotation
    root_rot = quat.normalize(
        quat.between(np.array([[0, 0, 1]]).repeat(len(root_fwd), axis=0), root_fwd)
    )

    # Find look at direction
    gaze_lookat = quat.mul_vec(grot[:, anim_data["names"].index("Head")], np.array([0, 0, 1]))
    gaze_lookat[:, 1] = 0
    gaze_lookat = gaze_lookat / np.sqrt(np.sum(np.square(gaze_lookat), axis=-1))[..., np.newaxis]
    # Find gaze position
    gaze_distance = 100  # Assume other actor is one meter away
    gaze_pos_all = root_pos + gaze_distance * gaze_lookat
    gaze_pos = np.median(gaze_pos_all, axis=0)
    gaze_pos = gaze_pos[np.newaxis].repeat(nframes, axis=0)

    # Visualize Gaze Pos
    visualize_gaze = False
    if visualize_gaze:
        import matplotlib.pyplot as plt

        plt.scatter(gaze_pos_all[:, 0], gaze_pos_all[:, 2], s=0.1, marker=".")
        plt.scatter(gaze_pos[0, 0], gaze_pos[0, 2])
        plt.scatter(root_pos[:, 0], root_pos[:, 2], s=0.1, marker=".")
        plt.quiver(root_pos[::60, 0], root_pos[::60, 2], root_fwd[::60, 0], root_fwd[::60, 2])
        plt.gca().set_aspect("equal")
        plt.savefig('1.jpg')
        plt.show()

    # Compute local gaze dir
    gaze_dir = gaze_pos - root_pos
    # gaze_dir = gaze_dir / np.sqrt(np.sum(np.square(gaze_dir), axis=-1))[..., np.newaxis]
    gaze_dir = quat.mul_vec(quat.inv(root_rot), gaze_dir)

    # Make relative to root
    lrot[:, 0] = quat.mul(quat.inv(root_rot), lrot[:, 0])
    lpos[:, 0] = quat.mul_vec(quat.inv(root_rot), lpos[:, 0] - root_pos)

    # Local velocities
    lvel = np.zeros_like(lpos)
    lvel[1:] = (lpos[1:] - lpos[:-1]) / dt
    lvel[0] = lvel[1] - (lvel[3] - lvel[2])

    lvrt = np.zeros_like(lpos)
    lvrt[1:] = quat.to_helical(quat.abs(quat.mul(lrot[1:], quat.inv(lrot[:-1])))) / dt
    lvrt[0] = lvrt[1] - (lvrt[3] - lvrt[2])

    # Root velocities
    root_vrt = np.zeros_like(root_pos)
    root_vrt[1:] = quat.to_helical(quat.abs(quat.mul(root_rot[1:], quat.inv(root_rot[:-1])))) / dt
    root_vrt[0] = root_vrt[1] - (root_vrt[3] - root_vrt[2])
    root_vrt[1:] = quat.mul_vec(quat.inv(root_rot[:-1]), root_vrt[1:])
    root_vrt[0] = quat.mul_vec(quat.inv(root_rot[0]), root_vrt[0])

    root_vel = np.zeros_like(root_pos)
    root_vel[1:] = (root_pos[1:] - root_pos[:-1]) / dt
    root_vel[0] = root_vel[1] - (root_vel[3] - root_vel[2])
    root_vel[1:] = quat.mul_vec(quat.inv(root_rot[:-1]), root_vel[1:])
    root_vel[0] = quat.mul_vec(quat.inv(root_rot[0]), root_vel[0])

    # Compute character space
    crot, cpos, cvrt, cvel = quat.fk_vel(lrot, lpos, lvrt, lvel, anim_data["parents"])

    # Compute 2-axis transforms
    ltxy = np.zeros(dtype=np.float32, shape=[len(lrot), njoints, 2, 3])
    ltxy[..., 0, :] = quat.mul_vec(lrot, np.array([1.0, 0.0, 0.0]))
    ltxy[..., 1, :] = quat.mul_vec(lrot, np.array([0.0, 1.0, 0.0]))

    ctxy = np.zeros(dtype=np.float32, shape=[len(crot), njoints, 2, 3])
    ctxy[..., 0, :] = quat.mul_vec(crot, np.array([1.0, 0.0, 0.0]))
    ctxy[..., 1, :] = quat.mul_vec(crot, np.array([0.0, 1.0, 0.0]))

    # return (
    #     root_pos,
    #     root_rot,
    #     root_vel,
    #     root_vrt,
    #     lpos,
    #     lrot,
    #     ltxy,
    #     lvel,
    #     lvrt,
    #     cpos,
    #     crot,
    #     ctxy,
    #     cvel,
    #     cvrt,
    #     gaze_pos,
    #     gaze_dir,
    # ), anim_data["parents"], dt, anim_data["order"]

    lpos = lpos.reshape(nframes, -1)
    ltxy = ltxy.reshape(nframes, -1)
    lvel = lvel.reshape(nframes, -1)
    lvrt = lvrt.reshape(nframes, -1)

    all_poses = np.concatenate((root_pos, root_rot, root_vel, root_vrt, lpos, ltxy, lvel, lvrt, gaze_dir), axis=1)

    return all_poses, anim_data["parents"], dt, anim_data["order"], njoints


def pose2bvh(poses, outpath, length, smoothing=False, smooth_foot=False):
    parents = np.array([-1,  0,  1,  2,  3,  4,  5,  6,  7,  4,  9, 10, 11, 12, 13, 14, 15,
       12, 17, 18, 19, 12, 21, 22, 23, 12, 25, 26, 27, 12, 29, 30, 31, 12,
       11,  4, 35, 36, 37, 38, 39, 40, 41, 38, 43, 44, 45, 38, 47, 48, 49,
       38, 51, 52, 53, 38, 55, 56, 57, 38, 37,  0, 61, 62, 63, 64, 63, 62,
        0, 68, 69, 70, 71, 70, 69], dtype=np.int32)
    order = 'zyx'
    dt = 0.05
    njoints = 75

    # smoothing
    if smoothing:
        n_poses = poses.shape[0]
        out_poses = np.zeros((n_poses, poses.shape[1]))
        for i in range(out_poses.shape[1]):
            # if (13 + (njoints - 14) * 9) <= i < (13 + njoints * 9): out_poses[:, i] = savgol_filter(poses[:, i], 41, 2)  # NOTE: smoothing on rotation matrices is not optimal
            # else:
            out_poses[:, i] = savgol_filter(poses[:, i], 15, 2)  # NOTE: smoothing on rotation matrices is not optimal
    else:
        out_poses = poses

    # Extract predictions
    P_root_pos = out_poses[:, 0:3]
    P_root_rot = out_poses[:, 3:7]
    P_root_vel = out_poses[:, 7:10]
    P_root_vrt = out_poses[:, 10:13]
    P_lpos = out_poses[:, 13 + njoints * 0: 13 + njoints * 3].reshape([length, njoints, 3])
    P_ltxy = out_poses[:, 13 + njoints * 3: 13 + njoints * 9].reshape([length, njoints, 2, 3])
    P_lvel = out_poses[:, 13 + njoints * 9: 13 + njoints * 12].reshape([length, njoints, 3])
    P_lvrt = out_poses[:, 13 + njoints * 12: 13 + njoints * 15].reshape([length, njoints, 3])

    P_ltxy = torch.as_tensor(P_ltxy, dtype=torch.float32)
    P_lrot = quat.from_xform(txform.xform_orthogonalize_from_xy(P_ltxy).cpu().numpy())        #

    if smooth_foot:
        pdb.set_trace()
        next_poses_LeftToeBase = P_lrot[:, -7]      # (length, 4)       7/14, 5/12
        next_poses_RightToeBase = P_lrot[:, -14]
        next_poses_LeftToeBase = np.zeros_like(next_poses_LeftToeBase)
        next_poses_RightToeBase = np.zeros_like(next_poses_RightToeBase)
        P_lrot[:, -7] = next_poses_LeftToeBase
        P_lrot[:, -14] = next_poses_RightToeBase

    # 20fps -> 60fps
    dt = 1 / 60
    P_root_pos = P_root_pos.repeat(3, axis=0)
    P_root_rot = P_root_rot.repeat(3, axis=0)
    P_lpos = P_lpos.repeat(3, axis=0)
    P_lrot = P_lrot.repeat(3, axis=0)

    write_bvh(outpath,
              P_root_pos,
              P_root_rot,
              P_lpos,
              P_lrot,
              parents, bone_names, order, dt
              )

if __name__ == '__main__':
    '''
    cd mymdm/process
    python process_zeggs_bvh.py
    '''
    config_file = "../../ubisoft-laforge-ZeroEGGS-main/configs/data_pipeline_conf_v1.json"
    with open(config_file, "r") as f:
        conf = json.load(f)

    conf = DictConfig(conf)

    # animation_file = "../../ubisoft-laforge-ZeroEGGS-main/Data/original/001_Neutral_0.bvh"
    animation_file = r"E:\下载\bvh2fpx"

    # (
    #     root_pos,       # (8116, 3)
    #     root_rot,       # (8116, 4)
    #     root_vel,       # (8116, 3)     # 1
    #     root_vrt,       # (8116, 3)     # 2
    #     lpos,       # (8116, 75, 3)     # 3
    #     lrot,       # (8116, 75, 4)
    #     ltxy,       # (8116, 75, 2, 3)      # 4
    #     lvel,       # (8116, 75, 3)     # 5
    #     lvrt,       # (8116, 75, 3)     # 6
    #     cpos,       # (8116, 75, 3)
    #     crot,       # (8116, 75, 4)
    #     ctxy,       # (8116, 75, 2, 3)
    #     cvel,       # (8116, 75, 3)
    #     cvrt,       # (8116, 75, 3)
    #     gaze_pos,       # (8116, 3)
    #     gaze_dir,       # (8116, 3)     # 7
    # ), parents, dt, order = preprocess_animation(animation_file)
    for item in os.listdir(os.path.join(animation_file, '20fps')):
        print(item)
        all_poses, parents, dt, order, njoints = preprocess_animation(os.path.join(animation_file, '20fps', item), fps=60)       # 20
        pose2bvh(poses=all_poses, outpath=os.path.join(animation_file, 'processed', item), length=all_poses.shape[0], smoothing=True, smooth_foot=False)

    # length = all_poses.shape[0]

    # root_rot = torch.as_tensor(root_rot, dtype=torch.float32)
    # gaze_pos = torch.as_tensor(gaze_pos, dtype=torch.float32)
    # root_pos = torch.as_tensor(root_pos, dtype=torch.float32)

    # pose_mean = np.load(r'E:\下载\mean.npz')['mean']
    # pose_std = np.load(r'E:\下载\std.npz')['std']
    # # normalize
    # std = np.clip(pose_std, a_min=0.01, a_max=None)
    # all_poses = (all_poses - pose_mean) / std
    # np.savez(r"E:\下载\bvh\happy-normalize.npz", pose=all_poses)
    # out_poses = all_poses
    # # denormalize
    # out_poses = np.multiply(out_poses, std) + pose_mean
    #
    # outpath = "../mydiffusion_zeggs/sample_20230104_192239/20230104_193613_smoothing_SG_minibatch_2720_[1, 0, 0, 0, 0, 0]_123456_1.bvh"
    # pose2bvh(poses=out_poses, outpath="../mydiffusion_zeggs/sample_20230104_192239/20230104_193613_smoothing_SG_minibatch_2720_[1, 0, 0, 0, 0, 0]_123456_1.bvh", length=length, smoothing=True, smooth_foot=False)



    # root_vel_mean = root_vel.mean(axis=0)
    # root_vrt_mean = root_vrt.mean(axis=0)
    # lpos_mean = lpos.mean(axis=0)
    # ltxy_mean = ltxy.mean(axis=0)
    # lvel_mean = lvel.mean(axis=0)
    # lvrt_mean = lvrt.mean(axis=0)
    # gaze_dir_mean = gaze_dir.mean(axis=0)
    #
    # anim_mean = np.hstack([root_vel_mean.ravel(), root_vrt_mean.ravel(), lpos_mean.ravel(), ltxy_mean.ravel(), lvel_mean.ravel(), lvrt_mean.ravel(), gaze_dir_mean.ravel()])
    #
    # root_vel_std = root_vel.std() + 1e-10
    # root_vrt_std = root_vrt.std() + 1e-10
    # lpos_std = lpos.std() + 1e-10
    # ltxy_std = ltxy.std() + 1e-10
    # lvel_std = lvel.std() + 1e-10
    # lvrt_std = lvrt.std() + 1e-10
    # gaze_dir_std = gaze_dir.std() + 1e-10
    #
    # anim_input_std = np.hstack([root_vel_std.repeat(len(root_vel_mean.ravel())),
    #                       root_vrt_std.repeat(len(root_vrt_mean.ravel())),
    #                       lpos_std.repeat(len(lpos_mean.ravel())),
    #                       ltxy_std.repeat(len(ltxy_mean.ravel())),
    #                       lvel_std.repeat(len(lvel_mean.ravel())),
    #                       lvrt_std.repeat(len(lvrt_mean.ravel())),
    #                       gaze_dir_std.repeat(len(gaze_dir_mean.ravel()))])
    #
    # root_vel_std = root_vel.std(axis=0)
    # root_vrt_std = root_vrt.std(axis=0)
    # lpos_std = lpos.std(axis=0)
    # ltxy_std = ltxy.std(axis=0)
    # lvel_std = lvel.std(axis=0)
    # lvrt_std = lvrt.std(axis=0)
    # gaze_dir_std = gaze_dir.std(axis=0)
    #
    # anim_output_std = np.hstack([root_vel_std.ravel(),
    #                       root_vrt_std.ravel(),
    #                       lpos_std.ravel(),
    #                       ltxy_std.ravel(),
    #                       lvel_std.ravel(),
    #                       lvrt_std.ravel(),
    #                       gaze_dir_std.ravel()])
    #
    # Z_root_vel = torch.as_tensor(root_vel, dtype=torch.float32)
    # Z_root_vrt = torch.as_tensor(root_vrt, dtype=torch.float32)
    # Z_lpos = torch.as_tensor(lpos, dtype=torch.float32)
    # Z_ltxy = torch.as_tensor(ltxy, dtype=torch.float32)
    # Z_lvel = torch.as_tensor(lvel, dtype=torch.float32)
    # Z_lvrt = torch.as_tensor(lvrt, dtype=torch.float32)
    # # gaze_dir = torch.as_tensor(gaze_dir, dtype=torch.float32)       #
    #
    # # Compute Local Gaze
    # Z_gaze_dir = tquat.quat_inv_mul_vec(root_rot, gaze_pos - root_pos)
    #
    #
    #
    # pose_encoding = torch.cat(
    #     [
    #         Z_root_vel.reshape([length, -1]),
    #         Z_root_vrt.reshape([length, -1]),
    #         Z_lpos.reshape([length, -1]),
    #         Z_ltxy.reshape([length, -1]),
    #         Z_lvel.reshape([length, -1]),
    #         Z_lvrt.reshape([length, -1]),
    #         Z_gaze_dir.reshape([length, -1]),
    #     ],
    #     dim=1,
    # )       # Need to Normalize
    #
    # pdb.set_trace()
    # pose_encoding = (pose_encoding - anim_mean) / anim_input_std
    #

    #
    # # processed_data_path = "../../ubisoft-laforge-ZeroEGGS-main/Data/processed_v1/processed_data.npz"
    # # processed_data = np.load(processed_data_path)
    # pdb.set_trace()
