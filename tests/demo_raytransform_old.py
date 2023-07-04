from fastatomography.util import *
n = 100
nxyz = (n,int(n*1.2),int(n*1.2))
projection_shape = (nxyz[0], nxyz[1])
d = 20
real_space_extent = [d, d*1.2, d*1.2]
n_angles = 20


def ray_transforms(real_space_extent, projection_shape, angles_in, interp='linear'):
    """
    Generate the ASTRA-based ray-projection and ray-back-projection operators
    :param real_space_extent: array (3,) the number of pixels in the three volume reconstruction dimensions
    :param projection_shape: array or tuple (2,) shape of the projections
    :param num_projections: int, number of projections to calculate
    :param interp: string, 'nearest' or 'linear'
    :return: A and At the Raytransform ind its agjoint
    call it like

    projections = A(volume, out=projections, angles=angles)

    where volume is a torch cuda tensor of shape (projection_shape[0], projection_shape[1], projection_shape[1])
          projections is a torch cuda tensor of shape projection_shape
          angles is a numpy array of shape (3, num_projections)
    """
    assert len(real_space_extent) == 3, "len(real_space_extent) != 3"
    assert len(projection_shape) == 2, "len(projection_shape) != 2"
    import numpy as np
    from fastatomography.tomo import RayTransform, RayBackProjection, Parallel3dEulerGeometry
    import odl

    num_projections = angles_in.shape[0]

    reco_space = odl.uniform_discr(
        min_pt=[-real_space_extent[0] / 2, -real_space_extent[1] / 2, -real_space_extent[2] / 2],
        max_pt=[real_space_extent[0] / 2, real_space_extent[1] / 2, real_space_extent[2] / 2],
        shape=[projection_shape[0], projection_shape[1], projection_shape[1]],
        dtype='float32', interp=interp)

    phi = np.linspace(0, np.deg2rad(90), int(np.ceil(num_projections ** (1 / 3))))
    theta = np.linspace(0, np.deg2rad(0.5), int(np.ceil(num_projections ** (1 / 3))))
    psi = np.linspace(0, np.deg2rad(0.5), int(np.ceil(num_projections ** (1 / 3))))
    angle_partition = odl.nonuniform_partition(phi, theta, psi)
    print('angle_partition', angle_partition)
    print()
    detector_partition = odl.uniform_partition([-real_space_extent[0] / 2, -real_space_extent[1] / 2],
                                               [real_space_extent[0] / 2, real_space_extent[1] / 2],
                                               [projection_shape[0], projection_shape[1]])
    print('detector_partition', detector_partition)
    print()
    geometry = Parallel3dEulerGeometry(angle_partition, detector_partition, check_bounds=False)
    print('geometry', geometry)
    print()
    angle_partition_dummy = odl.uniform_partition(
        min_pt=[angles_in.min(), -real_space_extent[0] / 2, -real_space_extent[1] / 2],
        max_pt=[angles_in.max(), real_space_extent[0] / 2, real_space_extent[1] / 2],
        shape=[num_projections, projection_shape[0], projection_shape[1]])
    print('angle_partition_dummy', angle_partition_dummy)
    print()
    domain = odl.uniform_discr_frompartition(angle_partition_dummy, dtype=np.float32)
    ray_trafo = RayTransform(reco_space, geometry, impl='astra_cuda')

    # proj_fspace = FunctionSpace(geometry.params, out_dtype=np.float32)
    # proj_space = DiscreteLp(
    #     proj_fspace, geometry.partition, proj_tspace,
    #     interp=proj_interp, axis_labels=axis_labels)
    #
    print('reco_space', reco_space)
    print('domain', domain)
    print()
    rayback_trafo = RayBackProjection(reco_space, geometry, impl='astra_cuda', domain=domain)
    return ray_trafo, rayback_trafo


# reco_space = odl.uniform_discr(
#     min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[n, n, n],
#     dtype='float32')
#
# phi = np.linspace(0, np.deg2rad(90), 3)
# theta = np.linspace(0, np.deg2rad(0.5), 3)
# psi = np.linspace(0, np.deg2rad(0.5), 3)
#
#%%
# from scipy.spatial.transform import Rotation as R
# rot = R.from_euler('yxz', [0,0,90], degrees=True)
# rot2 = rot.as_euler('YXZ')
# rot2
#%%
angles = th.zeros((3, n_angles))
angles[0, :] = 0
angles[1, :] = 0
angles[2, :] = 0#np.deg2rad(-90)

angle_nonzero = 0
angles[angle_nonzero, :] = th.linspace(np.deg2rad(-90), np.deg2rad(90), n_angles)



#%% XYZ intrinsic
# xyz extrinsic
# Rotation.from_euler('YXZ')
#
# angle_partition2 = odl.nonuniform_partition(phi, theta, psi)
# angle_partition_dummy = odl.uniform_partition(min_pt=[-20, -20, -20], max_pt=[20, 20, 20],
#                                               shape=[np.prod(angle_partition2.shape), n, n])
#
# detector_partition = odl.uniform_partition([-20, -20], [20, 20], [n, n])
# geometry = tomo.Parallel3dEulerGeometry(angle_partition2, detector_partition, check_bounds=False)
# range1 = odl.uniform_discr_frompartition(angle_partition_dummy, dtype=np.float32)
# %%
# ray_trafo = RayTransform(reco_space, geometry, impl='astra_cuda')
# rayback_trafo = RayBackProjection(range1, geometry, impl='astra_cuda', domain=reco_space)

A, AH = ray_transforms(real_space_extent, projection_shape, angles[angle_nonzero,:])
# %%
s = 20
s2 = 30
vol_shape = (projection_shape[0], projection_shape[1], projection_shape[1])

print(vol_shape)

m = np.array(vol_shape) // 3
mm = np.array(vol_shape) // 2
# Create a discrete Shepp-Logan phantom (modified version)
phantom = th.zeros(vol_shape, dtype=th.float32).cuda()
# phantom[mm[0] - s:mm[0] + s, m[1] - s:m[1] + s, m[2] - s:m[2] + s] = 1
off = 40
p = [n//2,n//2,n//2]
w = 1
phantom[p[0]-w:p[0]+w,p[1]-w:p[1]+w,p[2]-w:p[2]+w] = 1
p = [n//2,n//2,n//2+off]
w = 2
phantom[p[0]-w:p[0]+w,p[1]-w:p[1]+w,p[2]-w:p[2]+w] = 1
p = [n//2,n//2+off,n//2]
w = 3
phantom[p[0]-w:p[0]+w,p[1]-w:p[1]+w,p[2]-w:p[2]+w] = 1
p = [n//2+off,n//2,n//2]
w = 4
phantom[p[0]-w:p[0]+w,p[1]-w:p[1]+w,p[2]-w:p[2]+w] = 1
# phantom[m[0] - s2:m[0] + s2, m[1] - s2:m[1] + s2, m[2] - s2:m[2] + s2] = 0



#%%
reco = th.zeros(vol_shape, dtype=th.float).cuda()

# motion_shape = (np.prod(geometry.motion_partition.shape),)
# proj_shape = motion_shape + geometry.det_partition.shape
astra_proj_shape = (projection_shape[0], n_angles, projection_shape[1])

proj_data = th.zeros(astra_proj_shape, dtype=th.float).cuda()

# Create projection data by calling the ray transform on the phantom
proj_data = A(phantom, out=proj_data, angles=angles)
reco = AH(proj_data, out=reco, angles=angles)
#%%

# plot(th.sum(phantom,0).cpu().numpy())
# plot(th.sum(phantom,1).cpu().numpy())

# %%
for i in range(proj_data.shape[1]):
    plot(proj_data[:, i, :].cpu(),title=f'angle_nonzero: {angle_nonzero} {np.rad2deg(angles[:, i])}')
# %%
# for i in range(10):
#     plot(reco[:, 145 + i, :].cpu())
