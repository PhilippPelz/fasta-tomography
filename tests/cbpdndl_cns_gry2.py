#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This file is part of the SPORCO package. Details of the copyright
# and user license can be found in the 'LICENSE.txt' file distributed
# with the package.

"""
Convolutional Dictionary Learning
=================================

This example demonstrates the use of :class:`.dictlrn.cbpdndl.ConvBPDNDictLearn` for learning a convolutional dictionary
 from a set of training images. The dictionary learning algorithm is based on the ADMM consensus dictionary update
 :cite:`sorel-2016-fast` :cite:`garcia-2018-convolutional1`.
"""

from __future__ import print_function
from builtins import input
from builtins import range

import pyfftw  # See https://github.com/pyFFTW/pyFFTW/issues/40
import numpy as np

from sporco.dictlrn import cbpdndl
from sporco import util
from sporco import plot


"""
Load training images.
"""

path = '/home/philipp/projects2/tomo/2019-04-17-Pd_helix/philipp/'
S = np.load(path + 'helix_fista.npy')
S -= np.mean(S)
S = S[:, :, :]

ss = S.shape

"""
Construct initial dictionary.
"""

np.random.seed(12345)
D0 = np.random.randn(8, 8, 16)

"""
Set regularization parameter and options for dictionary learning solver.
"""
lmbda = 0.2
opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': 200,
                            'CBPDN': {'rho': 50.0*lmbda + 0.5},
                            'CCMOD': {'rho': 10.0, 'ZeroMean': True}},
                            dmethod='cns')

"""
Create solver object and solve.
"""
# %%
d = cbpdndl.ConvBPDNDictLearn(D0, S, lmbda, opt, dmethod='cns')
D1 = d.solve()
print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'))

"""
Display initial and final dictionaries.
"""
# %%
D1 = D1.squeeze()
fig = plot.figure(figsize=(14, 7))
plot.subplot(1, 2, 1)
plot.imview(util.tiledict(D0), title='D0', fig=fig)
plot.subplot(1, 2, 2)
plot.imview(util.tiledict(D1), title='D1', fig=fig)
fig.show()

"""
Get iterations statistics from solver object and plot functional value, ADMM primary and dual residuals, and automatically adjusted ADMM penalty parameter against the iteration number.
"""
# %%
its = d.getitstat()
fig = plot.figure(figsize=(20, 5))
plot.subplot(1, 3, 1)
plot.plot(its.ObjFun, xlbl='Iterations', ylbl='Functional', fig=fig)
plot.subplot(1, 3, 2)
plot.plot(np.vstack((its.XPrRsdl, its.XDlRsdl, its.DPrRsdl,
                     its.DDlRsdl)).T, ptyp='semilogy', xlbl='Iterations',
          ylbl='Residual', lgnd=['X Primal', 'X Dual', 'D Primal', 'D Dual'],
          fig=fig)
plot.subplot(1, 3, 3)
plot.plot(np.vstack((its.XRho, its.DRho)).T, xlbl='Iterations',
          ylbl='Penalty Parameter', ptyp='semilogy',
          lgnd=['$\\rho_X$', '$\\rho_D$'], fig=fig)
fig.show()

# %%
# Wait for enter on keyboard
# input()
D11 = D1.squeeze()
for i in range(D11.shape[3]):
    np.save(path + f"filter_{i}.npy", D11[:, :, :, i])
