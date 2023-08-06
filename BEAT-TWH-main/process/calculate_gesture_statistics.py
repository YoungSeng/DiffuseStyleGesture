import pdb
import argparse
import h5py
import numpy as np


def main(dataset, version):
    h5 = h5py.File(dataset + "_" + version + ".h5", "r")
    gesture_trn = [h5[key]['gesture'][:] for key in h5.keys()]
    h5.close()
    print("Total trn clips:", len(gesture_trn))     # Total trn clips: 27

    gesture_trn = np.vstack(gesture_trn)

    np.save("gesture_" + dataset + "_mean_" + version + ".npy", np.mean(gesture_trn, axis=0))
    np.save("gesture_" + dataset + "_std_" + version + ".npy", np.std(gesture_trn, axis=0) + 1e-6)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='calculate_gesture_statistics.py')
    parser.add_argument('--dataset', type=str, default='BEAT')
    parser.add_argument('--version', type=str, default='v0')
    args = parser.parse_args()
    main(args.dataset, args.version)
