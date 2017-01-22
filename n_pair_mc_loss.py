from chainer import cuda
from chainer.functions import matmul
from chainer.functions import transpose
from chainer.functions import softmax_cross_entropy
from chainer.functions import batch_l2_norm_squared


def n_pair_mc_loss(f, f_p, l2_reg):
    """Multi-class N-pair loss (N-pair-mc loss) function.

    Args:
        f (~chainer.Variable): Feature vectors.
            All examples must be different classes each other.
        f_p (~chainer.Variable): Positive examples corresponding to f.
            Each example must be the same class for each example in f.
        l2_reg (~float): A weight of L2 regularization for feature vectors.

    Returns:
        ~chainer.Variable: Loss value.

    See: `Improved Deep Metric Learning with Multi-class N-pair Loss \
        Objective <https://papers.nips.cc/paper/6200-improved-deep-metric-\
        learning-with-multi-class-n-pair-loss-objective>`_

    """
    logit = matmul(f, transpose(f_p))
    N = len(logit.data)
    xp = cuda.get_array_module(logit.data)
    loss_sce = softmax_cross_entropy(logit, xp.arange(N))
    l2_loss = sum(batch_l2_norm_squared(f) + batch_l2_norm_squared(f_p))
    loss = loss_sce + l2_reg * l2_loss
    return loss
