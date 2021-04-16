from fastatomography.util import *
from fastatomography.tomo import ray_transforms

n = 190
projection_shape = (n, n)
real_space_extent = np.array([n, n, n])
n_angles = 49

angles = th.zeros((3, n_angles))
angles[0, :] = 0
angles[1, :] = th.linspace(np.deg2rad(0), np.deg2rad(90), n_angles)
angles[2, :] = 0

A, AH = ray_transforms(real_space_extent, projection_shape, n_angles, interp='nearest')
# %%
from tomophantom import TomoP3D
from tomophantom.TomoP3D import Objects3D

# specify object parameters, here we replicate model
obj3D_1 = {'Obj': Objects3D.GAUSSIAN,
      'C0' : 1.0,
      'x0' :-0.25,
      'y0' : -0.15,
      'z0' : 0.0,
      'a'  : 0.3,
      'b'  :  0.2,
      'c'  :  0.3,
      'phi1'  : 35.0}

obj3D_2 = {'Obj': Objects3D.CUBOID,
      'C0' : 1.00,
      'x0' :0.1,
      'y0' : 0.2,
      'z0' : 0.0,
      'a'  : 0.15,
      'b'  :  0.35,
      'c'  :  0.6,
      'phi1'  : -60.0}

myObjects = [obj3D_1, obj3D_2] # dictionary of objects
phantom = th.from_numpy(TomoP3D.Object(n, myObjects)).float().cuda()
plot(np.sum(phantom.cpu().numpy(), 0))
plot(np.sum(phantom.cpu().numpy(), 1))
plot(np.sum(phantom.cpu().numpy(), 2))
#%%
s = 50
s2 = 30
vol_shape = (projection_shape[0], projection_shape[1], projection_shape[1])
m = np.array(vol_shape) // 2

reco = th.zeros(vol_shape, dtype=th.float).cuda()

astra_proj_shape = (projection_shape[0], n_angles, projection_shape[1])
proj_data = th.zeros(astra_proj_shape, dtype=th.float).cuda()

# Create projection data by calling the ray transform on the phantom
proj_data = A(phantom, out=proj_data, angles=angles)
reco = AH(proj_data, out=reco, angles=angles)
# %%
# plot(phantom[150, :, :])
# %%
for i in range(20,25):
    plot(proj_data[:, i, :].cpu(), f'proj {i} : angle {np.rad2deg(angles[1, i])}')
# %%
# for i in range(3):
#     plot(reco[:, 145 + i, :].cpu())
# #%%
# for i in range(3):
#     plot(phantom[:, 145 + i, :].cpu())
plot(proj_data[:, 0, :].cpu(), f'proj {i} : angle {np.rad2deg(angles[1, 0])}')
plot(proj_data[:, 10, :].cpu(), f'proj {i} : angle {np.rad2deg(angles[1, 10])}')
plot(proj_data[:, 20, :].cpu(), f'proj {i} : angle {np.rad2deg(angles[1, 20])}')