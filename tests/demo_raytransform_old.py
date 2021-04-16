from fastatomography.util import *

n = 300
projection_shape = (n, n)
real_space_extent = [20, 20, 20]
n_angles = 28


def ray_transforms(real_space_extent, projection_shape, num_projections, interp='nearest'):
    from fastatomography import tomo
    from fastatomography.tomo import RayTransform, RayBackProjection
    import odl

    reco_space = odl.uniform_discr(
        min_pt=[-real_space_extent[0] / 2, -real_space_extent[0] / 2, -real_space_extent[0] / 2],
        max_pt=[real_space_extent[0] / 2, real_space_extent[1] / 2, real_space_extent[2] / 2],
        shape=[projection_shape[0], projection_shape[1], projection_shape[1]],
        dtype='float32', interp=interp)
    angle_partition_dummy = odl.uniform_partition(
        min_pt=[-real_space_extent[0] / 2, -real_space_extent[0] / 2, -real_space_extent[0] / 2],
        max_pt=[real_space_extent[0] / 2, real_space_extent[1] / 2, real_space_extent[2] / 2],
        shape=[num_projections, projection_shape[0], projection_shape[1]])
    phi = np.linspace(0, np.deg2rad(90), int(np.ceil(num_projections ** (1 / 3))))
    theta = np.linspace(0, np.deg2rad(0.5), int(np.ceil(num_projections ** (1 / 3))))
    psi = np.linspace(0, np.deg2rad(0.5), int(np.ceil(num_projections ** (1 / 3))))
    angle_partition = odl.nonuniform_partition(phi, theta, psi)
    detector_partition = odl.uniform_partition([-real_space_extent[0] / 2, -real_space_extent[0] / 2],
                                               [real_space_extent[0] / 2, real_space_extent[0] / 2],
                                               [projection_shape[0], projection_shape[1]])
    geometry = tomo.Parallel3dEulerGeometry(angle_partition, detector_partition, check_bounds=False)
    range = odl.uniform_discr_frompartition(angle_partition_dummy, dtype=np.float32)
    ray_trafo = RayTransform(reco_space, geometry, impl='astra_cuda')
    rayback_trafo = RayBackProjection(range, geometry, impl='astra_cuda', domain=reco_space)
    return ray_trafo, rayback_trafo


# reco_space = odl.uniform_discr(
#     min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[n, n, n],
#     dtype='float32')
#
# phi = np.linspace(0, np.deg2rad(90), 3)
# theta = np.linspace(0, np.deg2rad(0.5), 3)
# psi = np.linspace(0, np.deg2rad(0.5), 3)
#
angles = th.zeros((3, n_angles))
angles[0, :] = 0
angles[1, :] = th.linspace(0, np.deg2rad(90), n_angles)
angles[2, :] = 0
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

A, AH = ray_transforms(real_space_extent, projection_shape, n_angles)
# %%
s = 50
s2 = 30
vol_shape = (projection_shape[0], projection_shape[1], projection_shape[1])
m = np.array(vol_shape) // 2
# Create a discrete Shepp-Logan phantom (modified version)
phantom = th.zeros(vol_shape, dtype=th.float32).cuda()
phantom[m[0] - s:m[0] + s, m[1] - s:m[1] + s, m[2] - s:m[2] + s] = 1
phantom[m[0] - s2:m[0] + s2, m[1] - s2:m[1] + s2, m[2] - s2:m[2] + s2] = 0

reco = th.zeros(vol_shape, dtype=th.float).cuda()

# motion_shape = (np.prod(geometry.motion_partition.shape),)
# proj_shape = motion_shape + geometry.det_partition.shape
astra_proj_shape = (projection_shape[0], n_angles, projection_shape[1])

proj_data = th.zeros(astra_proj_shape, dtype=th.float).cuda()

# Create projection data by calling the ray transform on the phantom
proj_data = A(phantom, out=proj_data, angles=angles)
reco = AH(proj_data, out=reco, angles=angles)
# %%
# plot(phantom[150, :, :].cpu())
# %%
for i in range(proj_data.shape[1]):
    plot(proj_data[:, i, :].cpu())
# %%
for i in range(10):
    plot(reco[:, 145 + i, :].cpu())
