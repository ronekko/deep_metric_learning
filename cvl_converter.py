# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:12:30 2015

@author: ryuhei
"""

import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import skimage.color
import skimage.filters
import skimage.morphology
import skimage.transform
import h5py
import fuel
from fuel.datasets.hdf5 import H5PYDataset
from progressbar import ProgressBar, Percentage, Bar, ETA, Counter


def opening(image, level):
    for i in range(level):
        image = skimage.morphology.erosion(image)
    for i in range(level):
        image = skimage.morphology.dilation(image)
    return image


def invert_black_and_white(image):
    dtype = image.dtype
    if dtype == np.uint8:
        white = 255
    elif dtype in (np.float16, np.float32, np.float64, np.float):
        white = 1.0
    else:
        raise ValueError("image.dtype must be uint8 or float")
    return white - image


def aabb(image, margin=0, return_bitmap=False):
    """
    returns axis aligned bounding box as boolean ndarray
    """
    hori = image.any(axis=0)
    vert = image.any(axis=1)

    height, width = image.shape
    left, right = hori.nonzero()[0][[0, -1]]
    top, bottom = vert.nonzero()[0][[0, -1]]
    new_left = np.maximum(0, left - margin)
    new_right = np.minimum(width, right + margin)
    new_top = np.maximum(0, top - margin)
    new_bottom = np.minimum(height, bottom + margin)

    ret = (new_left, new_right, new_top, new_bottom)
    if return_bitmap:
        hori[new_left:new_right] = True
        vert[new_top:new_bottom] = True
        bitmap = vert.reshape(-1, 1) * hori
        ret = ret + (bitmap,)
    return ret


def extract_writer_id(filename):
    """
    arg:
        filename: string, e.g. 'cvl-database-cropped-1-1/1122-4-cropped.tif'

    return:
        A tuple of writer id and text id, e.g. (1122, 4)
    """
    _, filename = filename.split("/")
    writer_id, text_id, _ = filename.split("-")
    return (int(writer_id), int(text_id))

def preprocess(image):
    gray = skimage.color.rgb2gray(image)
    gray_inv = invert_black_and_white(gray)
    threshold = skimage.filters.threshold_otsu(gray_inv)
    binary = gray_inv > threshold
    binary = opening(binary, 2)
    left, right, top, bottom = aabb(binary, 50)
    cropped = gray_inv[top:bottom, left:right]
    half_resolution = skimage.transform.rescale(cropped, 0.5)
    chw_shaped = np.expand_dims(half_resolution, 0)
    result = (chw_shaped * 255).astype(np.uint8)
    return result


if __name__ == '__main__':
    fuel_data_path = fuel.config.config["data_path"]["yaml"]
    filenames = ["cvl-database-1-1.zip", "cvl-database-cropped-1-1.zip"]
    filepaths = [os.path.join(fuel_data_path, name) for name in filenames]
    cvl_filepath, cvl_cropped_filepath = filepaths
    exclusion = ["cvl-database-cropped-1-1/0431-3-cropped.tif",
                 "cvl-database-cropped-1-1/0431-4-cropped.tif"]
    with zipfile.ZipFile(cvl_cropped_filepath) as zf:
        tif_filenames = [fn for fn in zf.namelist() if fn.endswith(".tif")]
        # remove filenames in exclusion from tif_filenames
        tif_filenames = list(set(tif_filenames) - set(exclusion))
        tif_filenames.sort()
#        tif_filenames = tif_filenames[:10]  # TODO: remove this line
        num_examples = len(tif_filenames)

    hdf5_filename = "cvl.hdf5"
    hdf5_filepath = os.path.join(fuel_data_path, hdf5_filename)
    hdf5 = h5py.File(hdf5_filepath, mode="w")
    shapes = []
    writer_ids = []
    text_ids = []

    # store flattened images as ragged array
    dtype = h5py.special_dtype(vlen=np.dtype(np.uint8))
    ds_images = hdf5.create_dataset("images", (num_examples,), dtype=dtype)
    ds_images.dims[0].label = "batch"

    # use ProgressBar
    widgets = ["{}:".format(filenames[1]), " ",
               Counter(), "/{}".format(num_examples), Percentage(), " ",
               Bar(), " ", ETA()]
    progress_bar = ProgressBar(widgets=widgets, maxval=num_examples).start()

    with zipfile.ZipFile(cvl_cropped_filepath) as zf:
        for i, filename in enumerate(tif_filenames):
            with zf.open(filename) as f:
                raw_image = plt.imread(f)
            image = preprocess(raw_image)
            ds_images[i] = image.flatten()

            shapes.append(image.shape)
            writer_id, text_id = extract_writer_id(filename)
            writer_ids.append(writer_id)
            text_ids.append(text_id)

            progress_bar.update(i)
        progress_bar.finish()

    # store original shapes of the flattened images
    shapes = np.array(shapes).astype(np.int32)
    ds_shapes = hdf5.create_dataset("shapes", (num_examples, 3),
                                    dtype=np.int32)
    ds_shapes[...] = shapes
    ds_images.dims.create_scale(ds_shapes, "shapes")
    ds_images.dims[0].attach_scale(ds_shapes)

    # add semantic tags of axes
    ds_shape_labels = hdf5.create_dataset("shape_labels", (3,), dtype="S7")
    ds_shape_labels[...] = [
        label.encode("utf8") for label in ["channel", "height", "width"]]
    ds_images.dims.create_scale(ds_shape_labels, "shape_labels")
    ds_images.dims[0].attach_scale(ds_shape_labels)

    # store the targets (writer IDs)
    targets = np.array(writer_ids, np.int32).reshape(num_examples, 1)
    ds_targets = hdf5.create_dataset("targets", data=targets)
    ds_targets.dims[0].label = "batch"
    ds_targets.dims[1].label = "weiter_id"

    # store the text IDs
    text_ids = np.array(text_ids, np.int32).reshape(num_examples, 1)
    ds_text_ids = hdf5.create_dataset("text_ids", data=text_ids)
    ds_text_ids.dims[0].label = "batch"
    ds_text_ids.dims[1].label = "text_id"

    # specify the splits
    split_train, split_test = (0, 189), (189, num_examples)
    split_dict = dict(train=dict(images=split_train,
                                 targets=split_train,
                                 text_ids=split_train),
                      test=dict(images=split_test,
                                targets=split_test,
                                text_ids=split_test))
    hdf5.attrs["split"] = H5PYDataset.create_split_array(split_dict)

    hdf5.flush()
    hdf5.close()
