# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 13:54:09 2017

@author: sakurai
"""

import os
import zipfile
import tarfile
import subprocess
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import h5py
import fuel
from fuel.datasets.hdf5 import H5PYDataset
from tqdm import tqdm
import cv2


def preprocess(hwc_bgr_image, size):
    hwc_rgb_image = cv2.cvtColor(hwc_bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(hwc_rgb_image, (size))
    chw_image = np.transpose(resized, axes=(2, 0, 1))
    return chw_image


if __name__ == '__main__':
    dataset_name = "online_products"
    archive_basename = "Stanford_Online_Products"

    fuel_root_path = fuel.config.config["data_path"]["yaml"]
    fuel_data_path = os.path.join(fuel_root_path, dataset_name)
    extracted_dir_path = os.path.join(fuel_data_path, archive_basename)
    archive_filepath = extracted_dir_path + ".zip"
    train_list_path = os.path.join(extracted_dir_path, "Ebay_train.txt")
    test_list_path = os.path.join(extracted_dir_path, "Ebay_test.txt")

    # Extract CUB_200_2011.tgz if CUB_200_2011 directory does not exist
    if not os.path.exists(os.path.join(fuel_data_path, archive_basename)):
        print "Extracting zip file. It may take a few minutes..."
        with zipfile.ZipFile(archive_filepath, "r") as zf:
            zf.extractall(fuel_data_path)

    train_records = np.loadtxt(train_list_path, np.str, skiprows=1)
    train_labels = train_records[:, 1].astype(np.int)
    train_files = train_records[:, 3]
    test_records = np.loadtxt(test_list_path, np.str, skiprows=1)
    test_labels = test_records[:, 1].astype(np.int)
    test_files = test_records[:, 3]

    jpg_filenames = np.concatenate((train_files, test_files))
    class_labels = np.concatenate((train_labels, test_labels))
    num_examples = len(jpg_filenames)

    # open hdf5 file
    hdf5_filename = dataset_name + ".hdf5"
    hdf5_filepath = os.path.join(fuel_data_path, hdf5_filename)
    hdf5 = h5py.File(hdf5_filepath, mode="w")

    # store images
    image_size = (256, 256)
    array_shape = (num_examples, 3) + image_size
    ds_images = hdf5.create_dataset("images", array_shape, dtype=np.uint8)
    ds_images.dims[0].label = "batch"
    ds_images.dims[1].label = "channel"
    ds_images.dims[2].label = "height"
    ds_images.dims[3].label = "width"

    # write images to the disk
    for i, filename in tqdm(enumerate(jpg_filenames), total=num_examples,
                            desc=hdf5_filepath):
        raw_image = cv2.imread(os.path.join(extracted_dir_path, filename),
                               cv2.IMREAD_COLOR)  # BGR image
        assert raw_image is not None
        image = preprocess(raw_image, image_size)
        ds_images[i] = image

    # store the targets (class labels)
    targets = np.array(class_labels, np.int32).reshape(num_examples, 1)
    ds_targets = hdf5.create_dataset("targets", data=targets)
    ds_targets.dims[0].label = "batch"
    ds_targets.dims[1].label = "class_labels"

    # specify the splits (labels 1~11318 for train, 11319~22634 for test)
    test_head = len(train_files)
    split_train, split_test = (0, test_head), (test_head, num_examples)
    split_dict = dict(train=dict(images=split_train, targets=split_train),
                      test=dict(images=split_test, targets=split_test))
    hdf5.attrs["split"] = H5PYDataset.create_split_array(split_dict)

    hdf5.flush()
    hdf5.close()
