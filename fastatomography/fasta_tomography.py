import torch as th
from fastatomography.fasta_tomo_custom import fasta
from fastatomography.util import shrink_nonnegative
from fastatomography.util import *


def fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, angles, opts):
    def f(z):
        # plot(z[:, 0, :].cpu().numpy(), 'z[:,0,:]')
        return .5 * th.norm(z - y) ** 2

    def gradf(z):
        return z - y

    def g(x):
        return th.norm(x, 1)

    def proxg(x, t):
        return shrink_nonnegative(x, t * mu)

    assert x0.dtype == th.float32, 'x0 dtype must be float32'
    assert y.dtype == th.float32, 'y dtype must be float32'
    sol, out, opts = fasta(A, At, f, gradf, g, proxg, x0, y, angles, opts)
    return sol, out, opts
