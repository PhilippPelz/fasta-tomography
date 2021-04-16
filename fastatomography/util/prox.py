import torch as th


def shrink(x, tau):
    xx = th.stack([th.abs(x) - tau, th.zeros_like(x)])
    maxx, inds = th.max(xx, 0)
    return th.sign(x) * maxx


def shrink_nonnegative(x, tau):
    # xx = th.stack([th.abs(x) - tau, th.zeros_like(x)])
    # maxx, inds = th.max(xx, 0)
    # del xx
    # maxx[th.sign(x) < 0] = 0
    x = th.abs(x) - tau
    x[th.sign(x) < 0] = 0
    return x
