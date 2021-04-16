from fastatomography.util import *
from fastatomography.tomo import ray_transforms
import numpy as np
from fastatomography.fasta_tomography import fasta_tomography_nonnegative_shrink

path = '/home/philipp/projects/'
fn = 'fept.emd'
f = h5read(path + fn)
x = f['data']['tomography']['data']
x[x < 0.50] = 0
x = th.as_tensor(x).float().cuda()
# %%
xs = x.shape
projection_shape = xs[:2]
real_space_extent = np.array([xs[0], xs[1], xs[2]])
n_crowther_projections = (th.max(th.tensor(x.shape)) * np.pi).int().item()
n_angles = n_crowther_projections

angles = th.zeros((3, n_angles))
angles[0, :] = 0
angles[1, :] = th.linspace(0, np.deg2rad(180), n_angles)
angles[2, :] = 0

A, At = ray_transforms(real_space_extent, projection_shape, n_angles)
# %%
vol_shape = (projection_shape[0], projection_shape[1], projection_shape[1])
x0 = th.zeros(vol_shape, dtype=th.float).cuda()

astra_proj_shape = (projection_shape[0], n_angles, projection_shape[1])
y0 = th.zeros(astra_proj_shape, dtype=th.float).cuda()
y = A(x, out=y0, angles=angles)
#%%
plot(y[:,0,:].cpu())
plot(y[:,200,:].cpu())
plot(y[:,400,:].cpu())
#%%
mu = 1e-4
opts = Param()
opts.record_objective = True
opts.verbose = True
opts.string_header = '     '
opts.max_iters = 100
opts.tol = 1e-4
opts.accelerate = True
opts.adaptive = True

sol, out, opts = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, angles, opts)
#%%
plot(sol[128,:,:].cpu())
plot(x[128,:,:].cpu())