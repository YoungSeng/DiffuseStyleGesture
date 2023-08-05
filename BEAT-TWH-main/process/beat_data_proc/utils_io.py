# -*- coding: utf-8 -*-
# @Author: dkang
# @Date: 2022-03-17 17:06:18
# @Last Modified by:   dkang
# @Last Modified time: 2022-03-17 17:06:18

import os
import sys
import h5py
import time
import datetime
import numpy as np


def load_h5_dataset(filename, *, ds_name_list=None, verbose=True, parser=None):
    # ds for dataset
    if verbose:
        print("    loading data from\n\t{} ...".format(os.path.basename(filename)))
        t0 = time.time()
    assert os.path.isfile(filename), "cannot find: {}".format(filename)

    def load_dict(d):
        ds_dict = {}
        for item in d.keys():
            if ds_name_list is not None and item not in ds_name_list:
                continue
            if isinstance(d[item], h5py._hl.dataset.Dataset):
                ds_dict[item] = d[item][()]
                if parser is not None and item in parser:
                    ds_dict[item] = parser[item](ds_dict[item])
            elif isinstance(d[item], h5py._hl.group.Group):
                ds_dict[item] = load_dict(d[item])
        return ds_dict

    with h5py.File(filename, "r") as f:
        ds_dict = load_dict(f)
    if verbose:
        print("    Elapsed time: {:.2f} sec".format(time.time() - t0))
    return ds_dict


def save_dataset_into_h5(filename, ds_dict, *, ds_name_list=None, overwrite=True, store_metadata=False, verbose=True):
    # ds for dataset
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        if overwrite:
            os.remove(filename)
        else:
            print("Error. Already exists file:\n\t{}".format(filename))
            print("^^^^^^^^^^")
            sys.exit(1)
    if ds_name_list is None:
        # then store all the fields by default.
        ds_name_list = ds_dict.keys()

    def save_data(f, d):
        assert isinstance(d, dict)
        for key in d:
            if isinstance(d[key], dict):
                g = f.create_group(key)
                save_data(g, d[key])
            else:
                try:
                    if len(d[key]) > 0 and type(d[key][0]) is str:
                        d[key] = [s.encode("ascii", "ignore") for s in d[key]]
                    f.create_dataset(
                        key,
                        data=d[key],
                        shape=d[key].shape,
                        dtype=d[key].dtype,
                        chunks=True,
                        fletcher32=True,
                        compression="gzip",
                        compression_opts=4,
                    )
                except:
                    f.create_dataset(
                        key,
                        data=d[key],
                    )

    with h5py.File(filename, "w") as f:
        save_data(f, ds_dict)

        if store_metadata:
            metadata = {"Date": datetime.datetime.fromtimestamp(time.time()).isoformat(), "OS": os.name}
            f.attrs.update(metadata)

    if verbose:
        print("Successfully saved into {}".format(filename))
