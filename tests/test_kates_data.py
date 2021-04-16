from fastatomography.util import *
from fastatomography.tomo import ray_transforms
import numpy as np
from fastatomography.fasta_tomography import fasta_tomography_nonnegative_shrink

from scipy.io import loadmat

path = '/home/philipp/projects2/tomo/2019-09-09_kate_pd/04_tomo_no_support/'
fn = '2019-10-07_genfire.mat'
angles_fn = 'angles1.mat'

proj = loadmat(path + fn)['d']
angles_in = np.deg2rad(loadmat(path + angles_fn)['a'])
#%%
y = th.as_tensor(np.transpose(proj, (1,2,0))).contiguous().cuda()
angles_in = th.as_tensor(angles_in.T).squeeze().contiguous().cuda()


# %%
ps = proj.shape
projection_shape = ps[:2]
real_space_extent = np.array([ps[0], ps[1], ps[1]])
n_angles = angles_in.shape[0]
angles = th.zeros((3, n_angles))
angles[0, :] = 0
angles[1, :] = angles_in
angles[2, :] = 0

A, At = ray_transforms(real_space_extent, projection_shape, n_angles)
x0 = th.zeros(projection_shape[0], projection_shape[1], projection_shape[1], dtype=th.float32).cuda()

#%%
mu = 1e-2
opts = Param()
opts.record_objective = True
opts.verbose = True
opts.string_header = '     '
opts.max_iters = 100
opts.tol = 1e-6
opts.accelerate = True
opts.adaptive = True

sol, out, opts = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, angles, opts)
#%%
# plot(sol[160,:,:].cpu())
for i in range(160,170):
    plot(sol[:,i,:].cpu())
# plot(sol[:,:,160].cpu())
#%%
np.save(path+'fasta.npy', sol.cpu().numpy())