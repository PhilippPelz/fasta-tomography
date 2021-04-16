import torch as th
from fastatomography.fasta import fasta


def fasta_sparseLeastSquares(A, At, b, mu, x0, opts):
    def f(z):
        return .5 * th.norm(z - b) ** 2

    def grad(z):
        return z - b

    def g(x):
        return th.norm(x, 1) * mu

    def prox(x, t):
        return shrink(x, t * mu)

    sol, out, opts = fasta(A, At, f, grad, g, prox, x0, opts)
    return sol, out, opts

def shrink(x, tau):
    xx = th.stack([th.abs(x) - tau,th.zeros_like(x)])
    maxx, inds = th.max(xx,0)
    return th.sign(x) * maxx
