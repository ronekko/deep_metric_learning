import numpy as np
import chainer.functions as F


def squared_distance_matrix(X, Y=None):
    if Y is None:
        Y = X
    return F.sum(((X[:, None] - Y[None]) ** 2), -1)


def proxy_nca_loss(x, proxy, labels):
    """Proxy-NCA loss function.

    Args:
        x (:class:`~chainer.Variable`):
            L2 normalized anchor points whose shape is (B, D), where B is the
            batch size and D is the number of dimensions of feature vector.
        proxy (:class:`~chainer.Variable` or :class:`~chainer.Parameter`):
            Proxies whose shape is (K, D), where K is the number of classes
            in the dataset.
        labels (:class:`numpy.ndarray`):
            Class labels associated to x. The shape is (B,) and dtype is int.
            Note that the class IDs must be 0, 1, ..., K-1.

    Returns:
        :class:`~chainer.Variable`: Loss value.

    See: `No Fuss Distance Metric Learning using Proxies \
        <http://openaccess.thecvf.com/content_ICCV_2017/papers/\
        Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.pdf>`_
    """
    proxy = F.normalize(proxy)
    distance = squared_distance_matrix(x, proxy)
    d_posi = distance[np.arange(len(x)), labels]

    # For each row, remove one element corresponding to the positive distance
    B, K = distance.shape  # batch size and the number of classes
    mask = np.tile(np.arange(K), (B, 1)) != labels[:, None]
    d_nega = distance[mask].reshape(B, K - 1)

    log_denominator = F.logsumexp(-d_nega, axis=1)
    loss = d_posi + log_denominator
    return F.average(loss)
