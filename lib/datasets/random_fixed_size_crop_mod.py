# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:29:34 2016

@author: sakurai
"""

from __future__ import division
import random

import numpy

try:
    from fuel.transformers._image import window_batch_bchw
    window_batch_bchw_available = True
except ImportError:
    window_batch_bchw_available = False
from fuel.transformers import ExpectsAxisLabels, SourcewiseTransformer
from fuel import config


class RandomFixedSizeCrop(SourcewiseTransformer, ExpectsAxisLabels):
    """Randomly crop images to a fixed window size.

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream`
        The data stream to wrap.
    window_shape : tuple
        The `(height, width)` tuple representing the size of the output
        window.
    random_lr_flip : bool
        If set to True, then images are horizontally flipped at random with a
        probability of 0.5, otherwise do nothing.
    devide_by_255 : bool
        If set to True, then images are normalized to [0, 1) as dtype float32.
    center_crop : bool
        If set to True, then images are center-cropped deterministically.

    Notes
    -----
    This transformer expects to act on stream sources which provide one of

     * Single images represented as 3-dimensional ndarrays, with layout
       `(channel, height, width)`.
     * Batches of images represented as lists of 3-dimensional ndarrays,
       possibly of different shapes (i.e. images of differing
       heights/widths).
     * Batches of images represented as 4-dimensional ndarrays, with
       layout `(batch, channel, height, width)`.

    The format of the stream will be un-altered, i.e. if lists are
    yielded by `data_stream` then lists will be yielded by this
    transformer.

    """
    def __init__(self, data_stream, window_shape=(224, 224),
                 random_lr_flip=False, devide_by_255=False, center_crop=False,
                 **kwargs):
        if not window_batch_bchw_available:
            raise ImportError('window_batch_bchw not compiled')
        self.window_shape = window_shape
        self.random_lr_flip = random_lr_flip
        self.devide_by_255 = devide_by_255
        self.center_crop = center_crop
        self.rng = kwargs.pop('rng', None)
        self.warned_axis_labels = False
        if self.rng is None:
            self.rng = numpy.random.RandomState(config.default_seed)
        kwargs.setdefault('produces_examples', data_stream.produces_examples)
        kwargs.setdefault('axis_labels', data_stream.axis_labels)
        super(RandomFixedSizeCrop, self).__init__(data_stream, **kwargs)

    def transform_source_batch(self, source, source_name):
        self.verify_axis_labels(('batch', 'channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        windowed_height, windowed_width = self.window_shape

        if (isinstance(source, list) or
            (isinstance(source, numpy.ndarray) and source.ndim == 1)) and all(
                isinstance(b, numpy.ndarray) and b.ndim == 3 for b in source):
            examples = [self.transform_source_example(im, source_name)
                        for im in source]
            if isinstance(source, list):
                return examples
            else:
                return numpy.array(examples)
        elif isinstance(source, numpy.ndarray) and source.ndim == 4:
            # Hardcoded assumption of (batch, channels, height, width).
            # This is what the fast Cython code supports.
            batch_size = source.shape[0]
            image_height, image_width = source.shape[2:]

            if self.center_crop:  # deterministic center crop
                offset_y = (image_height - windowed_height) // 2
                offset_x = (image_width - windowed_width) // 2
                out = source[:, :, offset_y:-offset_y, offset_x:-offset_x]
            else:  # random crop
                out = numpy.empty(source.shape[:2] + self.window_shape,
                                  dtype=source.dtype)
                max_h_off = image_height - windowed_height
                max_w_off = image_width - windowed_width
                if max_h_off < 0 or max_w_off < 0:
                    raise ValueError(
                        "Got ndarray batch with image dimensions {} but "
                        "requested window shape of {}".format(
                            source.shape[2:], self.window_shape))
                offsets_w = self.rng.random_integers(0, max_w_off,
                                                     size=batch_size)
                offsets_h = self.rng.random_integers(0, max_h_off,
                                                     size=batch_size)
                window_batch_bchw(source, offsets_h, offsets_w, out)

            if self.random_lr_flip:
                for example in out:
                    if random.randint(0, 1):
                        example[:] = example[:, :, ::-1]

            out = out.astype(numpy.float32)

            if self.devide_by_255:
                out = out.astype(numpy.float32) / 255.0

            return out
        else:
            raise ValueError("uninterpretable batch format; expected a list "
                             "of arrays with ndim = 3, or an array with "
                             "ndim = 4")

    def transform_source_example(self, example, source_name):
        self.verify_axis_labels(('channel', 'height', 'width'),
                                self.data_stream.axis_labels[source_name],
                                source_name)
        windowed_height, windowed_width = self.window_shape
        if not isinstance(example, numpy.ndarray) or example.ndim != 3:
            raise ValueError("uninterpretable example format; expected "
                             "ndarray with ndim = 3")
        image_height, image_width = example.shape[1:]
        if image_height < windowed_height or image_width < windowed_width:
            raise ValueError("can't obtain ({}, {}) window from image "
                             "dimensions ({}, {})".format(
                                 windowed_height, windowed_width,
                                 image_height, image_width))

        if self.center_crop:  # deterministic center crop
            off_h = (image_height - windowed_height) // 2
            off_w = (image_width - windowed_width) // 2
        else:  # random crop
            if image_height - windowed_height > 0:
                off_h = self.rng.random_integers(
                    0, image_height - windowed_height)
            else:
                off_h = 0
            if image_width - windowed_width > 0:
                off_w = self.rng.random_integers(
                    0, image_width - windowed_width)
            else:
                off_w = 0
        example = example[:, off_h:off_h + windowed_height,
                          off_w:off_w + windowed_width]

        # flip horizontally with a probability of 0.5
        if self.random_lr_flip:
            if random.randint(0, 1):
                example = example[:, :, ::-1]

        if self.devide_by_255:
            example = example.astype(numpy.float32) / 255.0

        return example
