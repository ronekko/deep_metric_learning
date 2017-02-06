import collections

import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check
from chainer import variable


def slices_to_indexes(slices, shape):
    """Get a flat 1D-array of integer indexes which is specified by the given
    slices to the shape.
    """
    if not isinstance(slices, tuple):
        slices = (slices,)
    index_subtensor = numpy.arange(numpy.prod(shape)).reshape(shape)
    none_slice = slice(None, None, None)  # [:], slice of end-to-end
    for d, slice_ in enumerate(slices):
        dth_indexes = (none_slice,) * d + (slice_, )
        index_subtensor = index_subtensor[dth_indexes]
    return index_subtensor.ravel()


class GetItem(function.Function):

    """Function that slices array and extract elements."""

    def __init__(self, slices):
        if isinstance(slices, (list, numpy.ndarray)):
            # TODO: if slices is list, confirm it is not nested
            slices = numpy.asarray(slices)
            if not slices.ndim == 1:
                raise ValueError('Advanced indexing currently supports '
                                 'only one dimensional list or ndarray')
        if not isinstance(slices, tuple):
            slices = (slices,)

        self._basic_indexing = True
        for s in slices:
            if isinstance(s, (list, numpy.ndarray)):
                self._basic_indexing = False
                break
        self.slices = slices

        if chainer.is_debug():
            n_ellipses = 0
            for s in slices:
                if numpy.isscalar(s) or s is None or isinstance(s,
                        (slice, list, numpy.ndarray)):
                    pass
                elif s is Ellipsis:
                    n_ellipses += 1
                else:
                    raise ValueError('Only basic indexing is supported')
            if n_ellipses > 1:
                raise ValueError('Only one Ellipsis is allowed')

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        if isinstance(self.slices, numpy.ndarray):
            type_check.expect(in_types[0].ndim == 1)
        else:
            valid_slice = len(self.slices) - self.slices.count(None)
            type_check.expect(in_types[0].ndim >= valid_slice)

    def forward(self, xs):
        ary = xs[0]
        return utils.force_array(ary[self.slices]),

    def backward(self, xs, gys):
        x = xs[0]
        gy = gys[0]
        xp = cuda.get_array_module(x)
        print self._basic_indexing
        print self.slices
        if self._basic_indexing:
            gx = xp.zeros_like(x)
            gx[self.slices] = gy
        else:
            flat_indexes = xp.asarray(slices_to_indexes(self.slices, x.shape))
            gx = xp.bincount(flat_indexes, gy.ravel(), x.size)
            gx = gx.reshape(x.shape).astype(gy.dtype)
        return gx,


def get_item(x, slices):
    """Extract elements from array with specified shape, axes and offsets.

    Args:
        x (tuple of Variables):
            Variable to be sliced.

        slices (int, slice, None or Ellipsis or tuple of them, list of int):
            Basic slicing to slice a variable. It supports ``int``, ``slice``,
            ``newaxis`` (equivalent to ``None``) and ``Ellipsis``. It also
            supports integer array indexing with some restrictions. When both
            x and slices are flat (i.e. x.ndim is 1 and slices is not nested),
            advanced indexing is available.

    Returns:
        Variable: :class:`~chainer.Variable` object
            which contains sliced array of ``x``.

    .. note::

       See NumPy document for details of `indexing
       <http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.

    """
    return GetItem(slices)(x)


def install_variable_get_item():
    variable.Variable.__getitem__ = get_item
