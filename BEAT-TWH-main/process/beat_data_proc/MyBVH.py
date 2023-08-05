# -*- coding: utf-8 -*-

import os
import sys
import pdb
import numpy as np
from typing import List, Dict

from scipy.spatial.transform import Rotation

#from utils_io import load_h5_dataset, decode_str
from .utils_io import load_h5_dataset
from .dataloaders.pymo.parsers import BVHParser
from .dataloaders.pymo.writers import BVHWriter


def euler2mat(angles, euler_orders):
    assert angles.ndim == 3 and angles.shape[2] == 3, f"wrong shape: {angles.shape}"
    assert angles.shape[1] == len(euler_orders)

    nJoints = len(euler_orders)
    nFrames = len(angles)
    rot_mats = np.zeros((nFrames, nJoints, 3, 3), dtype=np.float32)

    for j in range(nJoints):
        # {‘X’, ‘Y’, ‘Z’} for intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic rotations
        R = Rotation.from_euler(euler_orders[j].upper(), angles[:, j, :], degrees=True)  # upper for intrinsic rotation
        rot_mats[:, j, :, :] = R.as_matrix()
    return rot_mats


class MyBVH:
    def __init__(self, motion, keep_end_site=True):
        self.motion = motion
        self.joint_names = list(motion.skeleton.keys())
        if not keep_end_site:
            self.joint_names = [joint_name for joint_name in self.joint_names if not joint_name.endswith("_Nub")]
        self.framerate = np.round(1 / motion.framerate)
        # print("self.framerate:",self.framerate)

    def get_parents(self):
        parents = []
        for skeleton_name in self.joint_names:
            parent_name = self.motion.skeleton[skeleton_name]["parent"]
            if parent_name is None:
                parents.append(-1)
            else:
                parents.append(self.joint_names.index(parent_name))
        return parents

    def get_offsets(self):
        offsets = np.zeros((len(self.joint_names), 3))
        for i, joint_name in enumerate(self.joint_names):
            offsets[i] = self.motion.skeleton[joint_name]["offsets"]
        return offsets

    def get_global_positions(self):
        frame_count = self.motion.values.shape[0]
        trans = np.zeros((frame_count, 3))
        channel_name = f"{self.motion.root_name}_Xposition"
        if channel_name in self.motion.values:
            trans[:, 0] = self.motion.values[channel_name]
        channel_name = f"{self.motion.root_name}_Yposition"
        if channel_name in self.motion.values:
            trans[:, 1] = self.motion.values[channel_name]
        channel_name = f"{self.motion.root_name}_Zposition"
        if channel_name in self.motion.values:
            trans[:, 2] = self.motion.values[channel_name]
        return trans

    def get_rotations(self):
        frame_count = self.motion.values.shape[0]
        rotation = np.zeros((frame_count, len(self.joint_names), 3))
        rot_mats = np.zeros((frame_count, len(self.joint_names), 3, 3))
        euler_order = []
        for i, skeleton_name in enumerate(self.joint_names):
            temp = ""

            j = 0
            for channel in self.motion.channel_names:
                if channel[0] == skeleton_name and channel[1].endswith("rotation"):
                    temp += channel[1][0]
                    rotation[:, i, j] = self.motion.values[f"{channel[0]}_{channel[1]}"]
                    j += 1
            if len(temp) == 0:
                temp = "XYZ"
            assert len(temp) == 3

            euler_order.append(temp)
            R = Rotation.from_euler(temp.upper(), rotation[:, i], degrees=True)  # upper for intrinsic rotation
            rot_mats[:, i] = R.as_matrix()
        return rotation, rot_mats, euler_order


def selected_joint_names_to_idx(names: List[str], selected_joint_names: List[str]) -> List[str]:
    results = []
    if isinstance(names, np.ndarray):
        names = names.tolist()
    for joint_name in selected_joint_names:
        results.append(names.index(joint_name))
    return results


def trans2trans_mat_np(trans):
    assert trans.shape[-1] == 3
    trans_mat = np.eye(4)
    trans_mat = np.tile(trans_mat, (*trans.shape[:-1], 1, 1))
    trans_mat[..., :3, 3] = trans
    return trans_mat


def select_joints(selected_joint_names, joint_names, *, parents, offsets=None, motion=None):
    selected_joint_idx = selected_joint_names_to_idx(joint_names, selected_joint_names)
    global_transmat = np.tile(np.identity(4), (len(joint_names), 1, 1))
    if motion is None:
        motion_ = None
    else:
        motion_ = motion[:, selected_joint_idx]
    if offsets is None:
        offsets = np.zeros((len(joint_names), 3))
    offsets_ = np.zeros((len(selected_joint_names), 3))
    parents_ = np.zeros((len(selected_joint_names),), dtype=np.int32)
    for selected_joint, parent_of_selected_joint in enumerate(parents[1:], 1):
        # calculate the joint position of default pose
        global_transmat[selected_joint] = global_transmat[parent_of_selected_joint] @ (
            trans2trans_mat_np(offsets[selected_joint])
        )
    for new_idx, selected_joint in enumerate(selected_joint_idx):
        parent_of_selected_joint = parents[selected_joint]
        while True:
            # find the nearest parent of the selected joint
            if parent_of_selected_joint == -1:
                # current joint is the root joint
                parents_[new_idx] = -1
                offsets_[new_idx] = global_transmat[selected_joint][:3, 3]
                break
            elif parent_of_selected_joint in selected_joint_idx:
                parent_new_idx = selected_joint_idx.index(parent_of_selected_joint)
                parents_[new_idx] = parent_new_idx
                # calculate the new offset of current joint
                temp = np.linalg.inv(global_transmat[parent_of_selected_joint]) @ global_transmat[selected_joint]
                offsets_[new_idx] = temp[:3, 3]
                break
            else:
                # the ancestor node is not selected
                # merge the rotation of the ancestor joint to current joint
                parent_of_selected_joint = parents[parent_of_selected_joint]
    return parents_, offsets_, motion_


def load_bvh_data(fn):
    parser = BVHParser()
    motion = parser.parse(fn)
    bvh = MyBVH(motion, keep_end_site=False)

    parents = bvh.get_parents()
    offsets = bvh.get_offsets()
    eulers, rot_mats, euler_orders = bvh.get_rotations()
    global_pos = bvh.get_global_positions()

    info = {}
    info["joint_names"] = bvh.joint_names
    info["offsets"] = offsets
    info["parents"] = np.asarray(parents, dtype=np.int32)
    info["euler_orders"] = euler_orders
    info["framerate"] = bvh.framerate

    info["rot_angles"] = eulers
    info["rot_mats"] = rot_mats
    info["global_pos"] = global_pos
    return info


def write_bvh_data(
    bvh_fn,
    *,
    joint_names,
    skeleton_tree,
    offsets,
    euler_orders,
    framerate=None,
    motion=None,
    global_trans=None,
    with_endsite=False,
):
    has_children = lambda i: i in skeleton_tree
    parser = BVHParser()

    for i, joint_name in enumerate(joint_names):
        parent_idx = skeleton_tree[i]
        orders = euler_orders[i]
        channels = [
            f"{orders[0]}rotation",
            f"{orders[1]}rotation",
            f"{orders[2]}rotation",
        ]
        if parent_idx == -1:
            parser.root_name = joint_name
            parent_name = None
            channels = ["Xposition", "Yposition", "Zposition"] + channels
        else:
            parent_name = joint_names[parent_idx]
            parser._skeleton[parent_name]["children"].append(joint_name)
        joint = parser._new_bone(parent=parent_name,name=None)
        joint["offsets"] = offsets[i].tolist()
        parser._skeleton[joint_name] = joint
        if (not with_endsite) and (not has_children(i)):
            endsite = parser._new_bone(joint_name,name=None)
            endsite["offsets"] = [0, 0, 0]
            parser._skeleton[f"{joint_name}_Nub"] = endsite
            parser._skeleton[joint_name]["children"].append(f"{joint_name}_Nub")
        if (with_endsite) and has_children(i) or (not with_endsite):
            joint["channels"] = channels
    for i, joint_name in enumerate(joint_names):
        if (with_endsite and has_children(i)) or (not with_endsite):
            orders = euler_orders[i]
            if i == 0:
                parser._motion_channels.append((joint_name, "Xposition"))
                parser._motion_channels.append((joint_name, "Yposition"))
                parser._motion_channels.append((joint_name, "Zposition"))
            parser._motion_channels.append((joint_name, f"{orders[0]}rotation"))
            parser._motion_channels.append((joint_name, f"{orders[1]}rotation"))
            parser._motion_channels.append((joint_name, f"{orders[2]}rotation"))

    if framerate is not None:
        framerate = 1 / framerate
        parser.framerate = framerate
    frame_count = motion.shape[0]
    if global_trans is None:
        global_trans = np.zeros((frame_count, 3))
    if with_endsite:
        joints_with_rotations = np.array([(not joint_name.endswith("Nub")) for joint_name in joint_names])
        motion = motion[:, joints_with_rotations]
    print("motion2:",motion[0,36,:])
    motion = motion.reshape(frame_count, -1)
    print("motion3:",motion[0,3+105:6+105])
    motion = np.concatenate((global_trans, motion), axis=1)
    parser._motions = [()] * frame_count
    frame_time = 0
    for j, channel in enumerate(parser._motion_channels):
        print(motion[0, j],"      ",channel[0],"      ",channel[1])


    for i in range(frame_count):
        channel_values = [
            (
                channel[0],
                channel[1],
                motion[i, j],
            )
            for j, channel in enumerate(parser._motion_channels)
        ]
        parser._motions[i] = (frame_time, channel_values)
        frame_time += framerate
    parser.data.skeleton = parser._skeleton
    parser.data.channel_names = parser._motion_channels
    parser.data.values = parser._to_DataFrame()
    parser.data.root_name = parser.root_name
    parser.data.framerate = parser.framerate

    writer = BVHWriter()
    with open(bvh_fn, "w") as f:
        writer.write(parser.data, f)


if __name__ == "__main__":
    # bvh_fn = r"E:\Downloads\1_wayne_0_1_1.bvh"
    # assert os.path.isfile(bvh_fn)
    # data_dict = load_bvh_data(bvh_fn)
    # pdb.set_trace()
    # sys.exit(0)
    mean_pose = np.load("/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/audio2pose/datasets/beat_cache/beat_4english_15_141train/bvh_rot/bvh_mean.npy")
    std_pose = np.load("/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/audio2pose/datasets/beat_cache/beat_4english_15_141train/bvh_rot/bvh_std.npy")

    h5_fn = r"/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/BEAT/beat_English_15FPS_75/test/data/2_scott_0_103_103.h5"
    #bvh_save_fn = r"E:\Downloads\1_wayne_0_1_1-from_h5.bvh"
    data = load_h5_dataset(h5_fn)
    euler_orders = data["euler_orders"]
    if "rot_angles" in data:
        euler_angles = data["rot_angles"]
    else:
        rot_mats = data["rot_mats"]
        euler_angles = np.zeros((rot_mats.shape[0], rot_mats.shape[1], 3))
        for i in range(rot_mats.shape[1]):
            euler_angles[:, i] = Rotation.from_matrix(rot_mats[:, i]).as_euler(euler_orders[i], degrees=True)
    '''
    global_pos = data.get("global_pos", None)
    data = write_bvh_data(
        bvh_save_fn,
        joint_names=data["joint_names"],
        offsets=data["offsets"],
        skeleton_tree=data["parents"],
        euler_orders=euler_orders,
        framerate=data["framerate"],
        motion=euler_angles,
        global_trans=global_pos,
        with_endsite=False,
    )
    '''
    '''
    info = load_bvh_data("/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/BEAT/beat_genea/examples/2_scott_1_1_1.bvh")
    print(info["euler_orders"][: 75])
    euler_orders=info["euler_orders"][: 75]
    pose_pred_npz=euler_angles.copy()
    #pose_pred_npz=pose_pred_npz * std_pose + mean_pose
    print("pose_pred_npz shape:",pose_pred_npz.shape)
    print(pose_pred_npz)
    pred_pose_rotmat=euler2mat(pose_pred_npz, euler_orders).astype(np.float32)#34,47,3,3
    np.savez("pred_rotmat_new1.npz",pred_pose_rotmat=pred_pose_rotmat)
    '''



    info = load_bvh_data("/apdcephfs/share_1290939/shaolihuang/ykcheng/MS_RP/AGDM/BEAT/beat_english_v0.2.1/1/1_wayne_0_2_2.bvh")
    nframes = len(info["rot_angles"])
    print("nframes1:",nframes)
    njoints = len(info["parents"])

    print("rotations:",info["rot_angles"][0])

