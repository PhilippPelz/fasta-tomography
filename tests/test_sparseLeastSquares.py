import torch as th
from fastatomography.util.parameters import Param
from tests.fasta_sparseLeastSquares import fasta_sparseLeastSquares
import matplotlib.pyplot as plt

M = 100
N = 1000
K = 10
mu = 1e-4
sigma = .01

x = th.zeros(N)
perm = th.randperm(N)
x[perm[1:K]] = 1


A = th.randn(M, N)
A = A / th.norm(A)

b = A @ x
b = b + sigma * th.randn(b.shape)

x0 = th.zeros(N)

opts = Param()
opts.record_objective = True
opts.verbose = True
opts.string_header = '     '
opts.max_iters = 3500
opts.tol = 1e-5
opts.accelerate = True

def A1(x):
    return A @ x

def A1t(x):
    return A.t() @ x

sol, out, opts = fasta_sparseLeastSquares(A1, A1t, b, mu, x0, opts)

f, ax = plt.subplots()
markerline, stemlines, baseline = plt.stem(
    x, linefmt='b')
markerline.set_markerfacecolor('b')
markerline, stemlines, baseline = plt.stem(
    sol, linefmt='r')
markerline.set_markerfacecolor('r')
plt.show()

f, ax = plt.subplots(2,2)
ax[0,0].semilogy(out.func_values)
ax[0,0].set_title('func_values')
ax[0,0].set_xlabel('# iterations')
ax[1,0].semilogy(out.residuals)
ax[1,0].set_title('residuals')
ax[1,0].set_xlabel('# iterations')
ax[0,1].semilogy(out.objective)
ax[0,1].set_title('objective')
ax[0,1].set_xlabel('# iterations')
plt.show()
