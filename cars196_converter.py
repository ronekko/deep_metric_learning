# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:34:25 2016

@author: sakurai
"""

import os
import tarfile
import subprocess
import numpy as np
from scipy.io import loadmat
import h5py
import fuel
from fuel.datasets.hdf5 import H5PYDataset
import cv2
from tqdm import tqdm


def preprocess(hwc_bgr_image, size):
    hwc_rgb_image = cv2.cvtColor(hwc_bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(hwc_rgb_image, (size))
    chw_image = np.transpose(resized, axes=(2, 0, 1))
    return chw_image


if __name__ == '__main__':
    dataset_name = "cars196"
    archive_basename = "car_ims"

    fuel_root_path = fuel.config.config["data_path"]["yaml"]
    fuel_data_path = os.path.join(fuel_root_path, dataset_name)
    image_filepath = os.path.join(fuel_data_path, archive_basename + ".tgz")
    label_filepath = os.path.join(fuel_data_path, "cars_annos.mat")

    # Extract car_ims.tgz if car_ims directory does not exist
    with tarfile.open(image_filepath, "r") as tf:
        jpg_filenames = [fn for fn in tf.getnames() if fn.endswith(".jpg")]
    jpg_filenames.sort()
    num_examples = len(jpg_filenames)  # ????
    if not os.path.exists(os.path.join(fuel_data_path, archive_basename)):
        subprocess.call(["tar", "zxvf", image_filepath.replace("\\", "/"),
                         "-C", fuel_data_path.replace("\\", "/"),
                         "--force-local"])

    # Extract class labels
    cars_annos = loadmat(label_filepath)
    annotations = cars_annos["annotations"].ravel()
    annotations = sorted(annotations, key=lambda a: str(a[0][0]))
    class_labels = []
    for annotation in annotations:
        class_label = int(annotation[5])
        class_labels.append(class_label)

    # open hdf5 file
    hdf5_filename = "cars196.hdf5"
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
        raw_image = cv2.imread(os.path.join(fuel_data_path, filename),
                               cv2.IMREAD_COLOR)  # BGR image
        image = preprocess(raw_image, image_size)
        ds_images[i] = image

    # store the targets (class labels)
    targets = np.array(class_labels, np.int32).reshape(num_examples, 1)
    ds_targets = hdf5.create_dataset("targets", data=targets)
    ds_targets.dims[0].label = "batch"
    ds_targets.dims[1].label = "class_labels"

    # specify the splits (labels 1~98 for train, 99~196 for test)
    test_head = class_labels.index(98)
    split_train, split_test = (0, test_head), (test_head, num_examples)
    split_dict = dict(train=dict(images=split_train, targets=split_train),
                      test=dict(images=split_test, targets=split_test))
    hdf5.attrs["split"] = H5PYDataset.create_split_array(split_dict)

    hdf5.flush()
    hdf5.close()
