# Copyright 2014-2018 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Backend for ASTRA using CUDA."""

from __future__ import print_function, division, absolute_import
from builtins import object
from multiprocessing import Lock
import numpy as np
from packaging.version import parse as parse_version
import torch as th

try:
    import astra

    ASTRA_CUDA_AVAILABLE = astra.astra.use_cuda()
except ImportError:
    ASTRA_CUDA_AVAILABLE = False

from odl.discr import DiscreteLp
from fastatomography.tomo.backends import (
    ASTRA_VERSION,
    astra_projection_geometry, astra_projection_geometry2, astra_volume_geometry, astra_projector)
from fastatomography.tomo.geometry import (
    Geometry, Parallel2dGeometry, Parallel3dAxisGeometry,
    ConeFlatGeometry)

__all__ = ('ASTRA_CUDA_AVAILABLE',
           'AstraCudaProjectorImpl', 'AstraCudaBackProjectorImpl')


class AstraCudaProjectorImpl(object):
    """Thin wrapper around ASTRA."""

    algo_id = None
    vol_id = None
    sino_id = None
    proj_id = None

    def __init__(self, geometry, reco_space, proj_space):
        """Initialize a new instance.

        Parameters
        ----------
        geometry : `Geometry`
            Geometry defining the tomographic setup.
        reco_space : `DiscreteLp`
            Reconstruction space, the space of the images to be forward
            projected.
        proj_space : `DiscreteLp`
            Projection space, the space of the result.
        """
        assert isinstance(geometry, Geometry)
        assert isinstance(reco_space, DiscreteLp)
        assert isinstance(proj_space, DiscreteLp)

        self.geometry = geometry
        self.reco_space = reco_space
        self.proj_space = proj_space
        self.proj_ptr = -1
        self.vol_ptr = -1
        self.last_angles = None

        # self.create_ids(proj_tensor, vol_tensor)

        # Create a mutually exclusive lock so that two callers cant use the
        # same shared resource at the same time.
        self._mutex = Lock()

    def call_forward(self, vol_data, proj_tensor, angles=None):
        """Run an ASTRA forward projection on the given data using the GPU.

        Parameters
        ----------
        vol_data : ``reco_space`` element
            Volume data to which the projector is applied.
        proj_tensor : ``proj_space`` element, optional
            Element of the projection space to which the result is written. If
            ``None``, an element in `proj_space` is created.

        Returns
        -------
        proj_tensor : ``proj_space`` element
            Projection data resulting from the application of the projector.
            If ``proj_tensor`` was provided, the returned object is a reference to it.
        """
        with self._mutex:
            have_new_angles = self.last_angles is not None and self.last_angles != angles
            if self.proj_ptr is None or self.vol_ptr is None or self.proj_ptr != proj_tensor.data_ptr() or self.vol_ptr != vol_data.data_ptr() or have_new_angles:
                self.geometry.implementation_cache = {}
                self.create_ids2(proj_tensor, vol_data, angles)

            # Run algorithm
            astra.algorithm.run(self.algo_id)

            # Copy result to host
            if self.geometry.ndim == 2:
                proj_tensor[:] = self.out_array

            # Fix scaling to weight by pixel size
            if isinstance(self.geometry, Parallel2dGeometry):
                # parallel2d scales with pixel stride
                proj_tensor *= 1 / float(self.geometry.det_partition.cell_volume)

            return proj_tensor

    def create_ids2(self, proj_tensor, vol_tensor, angles=None):
        """Create ASTRA objects."""
        # Create input and output arrays
        self.proj_ptr = proj_tensor.data_ptr()
        self.vol_ptr = vol_tensor.data_ptr()
        if self.geometry.motion_partition.ndim == 1:
            motion_shape = self.geometry.motion_partition.shape
        else:
            # Need to flatten 2- or 3-dimensional angles into one axis
            motion_shape = (np.prod(self.geometry.motion_partition.shape),)

        proj_shape = motion_shape + self.geometry.det_partition.shape
        proj_ndim = len(proj_shape)
        # print(f'proj_shape {proj_shape}')
        # print(f'proj_ndim {proj_ndim}')

        if proj_ndim == 2:
            astra_proj_shape = proj_shape
            astra_vol_shape = self.reco_space.shape
        elif proj_ndim == 3:
            # The `u` and `v` axes of the projection data are swapped,
            # see explanation in `astra_*_3d_geom_to_vec`.
            astra_proj_shape = (proj_shape[1], proj_shape[0], proj_shape[2])
            astra_vol_shape = self.reco_space.shape

        # print(f'astra_proj_shape {astra_proj_shape}')
        # if proj_tensor.shape != astra_proj_shape:
        #     raise RuntimeError(
        #         f"proj_tensor.shape != astra_proj_shape : proj_tensor.shape = {proj_tensor.shape}, astra_proj_shape = {astra_proj_shape}")
        if vol_tensor.shape != astra_vol_shape:
            raise RuntimeError(
                f"vol_tensor.shape != astra_vol_shape : vol_tensor.shape = {vol_tensor.shape}, astra_vol_shape = {astra_vol_shape}")

        self.in_array = vol_tensor
        self.out_array = proj_tensor

        z, y, x = vol_tensor.shape
        # print(f"vol_tensor.shape {vol_tensor.shape}")
        vol_stride = th.tensor(vol_tensor.stride()) * vol_tensor.storage().element_size()
        vol_link = astra.data3d.GPULink(vol_tensor.storage().data_ptr(), x, y, z, vol_stride[-2])

        z, y, x = proj_tensor.shape
        # print(f"proj_tensor.shape {proj_tensor.shape}")
        proj_stride = th.tensor(proj_tensor.stride()) * proj_tensor.storage().element_size()
        proj_data_link = astra.data3d.GPULink(proj_tensor.storage().data_ptr(), x, y, z, proj_stride[-2])

        # Create ASTRA data structures
        vol_geom = astra_volume_geometry(self.reco_space)
        proj_geom = astra_projection_geometry2(self.geometry, angles)
        self.vol_id = astra.data3d.link('-vol', vol_geom, vol_link)
        # self.vol_id = astra_data(vol_geom,
        #                          datatype='volume',
        #                          ndim=self.reco_space.ndim,
        #                          data=self.in_array,
        #                          allow_copy=False)
        self.sino_id = astra.data3d.link('-sino', proj_geom, proj_data_link)

        self.proj_id = astra_projector('linear', vol_geom, proj_geom, ndim=proj_ndim, impl='cuda')

        # self.sino_id = astra_data(proj_geom,
        #                           datatype='projection',
        #                           ndim=proj_ndim,
        #                           data=self.out_array,
        #                           allow_copy=False)

        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['VolumeDataId'] = self.vol_id
        cfg['ProjectionDataId'] = self.sino_id
        self.algo_id = astra.algorithm.create(cfg)
        # Create algorithm
        # self.algo_id = astra_algorithm(
        #     'forward', proj_ndim, self.vol_id, self.sino_id,
        #     proj_id=self.proj_id, impl='cuda')

    def create_ids(self, proj_tensor, vol_tensor):
        """Create ASTRA objects."""
        # Create input and output arrays
        self.proj_ptr = proj_tensor.data_ptr()
        self.vol_ptr = vol_tensor.data_ptr()
        if self.geometry.motion_partition.ndim == 1:
            motion_shape = self.geometry.motion_partition.shape
        else:
            # Need to flatten 2- or 3-dimensional angles into one axis
            motion_shape = (np.prod(self.geometry.motion_partition.shape),)

        proj_shape = motion_shape + self.geometry.det_partition.shape
        proj_ndim = len(proj_shape)
        print(f'proj_shape {proj_shape}')
        print(f'proj_ndim {proj_ndim}')

        if proj_ndim == 2:
            astra_proj_shape = proj_shape
            astra_vol_shape = self.reco_space.shape
        elif proj_ndim == 3:
            # The `u` and `v` axes of the projection data are swapped,
            # see explanation in `astra_*_3d_geom_to_vec`.
            astra_proj_shape = (proj_shape[1], proj_shape[0], proj_shape[2])
            astra_vol_shape = self.reco_space.shape

        print(f'astra_proj_shape {astra_proj_shape}')
        if proj_tensor.shape != astra_proj_shape:
            raise RuntimeError(
                f"proj_tensor.shape != astra_proj_shape : proj_tensor.shape = {proj_tensor.shape}, astra_proj_shape = {astra_proj_shape}")
        if vol_tensor.shape != astra_vol_shape:
            raise RuntimeError(
                f"vol_tensor.shape != astra_vol_shape : vol_tensor.shape = {vol_tensor.shape}, astra_vol_shape = {astra_vol_shape}")

        self.in_array = vol_tensor
        self.out_array = proj_tensor

        z, y, x = vol_tensor.shape
        print(f"vol_tensor.shape {vol_tensor.shape}")
        vol_stride = th.tensor(vol_tensor.stride()) * vol_tensor.storage().element_size()
        vol_link = astra.data3d.GPULink(vol_tensor.storage().data_ptr(), x, y, z, vol_stride[-2])

        z, y, x = proj_tensor.shape
        print(f"proj_tensor.shape {proj_tensor.shape}")
        proj_stride = th.tensor(proj_tensor.stride()) * proj_tensor.storage().element_size()
        proj_data_link = astra.data3d.GPULink(proj_tensor.storage().data_ptr(), x, y, z, proj_stride[-2])

        # Create ASTRA data structures
        vol_geom = astra_volume_geometry(self.reco_space)
        proj_geom = astra_projection_geometry(self.geometry)
        self.vol_id = astra.data3d.link('-vol', vol_geom, vol_link)
        # self.vol_id = astra_data(vol_geom,
        #                          datatype='volume',
        #                          ndim=self.reco_space.ndim,
        #                          data=self.in_array,
        #                          allow_copy=False)
        self.sino_id = astra.data3d.link('-sino', proj_geom, proj_data_link)

        self.proj_id = astra_projector('nearest', vol_geom, proj_geom,
                                       ndim=proj_ndim, impl='cuda')

        # self.sino_id = astra_data(proj_geom,
        #                           datatype='projection',
        #                           ndim=proj_ndim,
        #                           data=self.out_array,
        #                           allow_copy=False)

        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['VolumeDataId'] = self.vol_id
        cfg['ProjectionDataId'] = self.sino_id
        self.algo_id = astra.algorithm.create(cfg)
        # Create algorithm
        # self.algo_id = astra_algorithm(
        #     'forward', proj_ndim, self.vol_id, self.sino_id,
        #     proj_id=self.proj_id, impl='cuda')

    def __del__(self):
        """Delete ASTRA objects."""
        if self.geometry.ndim == 2:
            adata, aproj = astra.data2d, astra.projector
        else:
            adata, aproj = astra.data3d, astra.projector3d

        if self.algo_id is not None:
            astra.algorithm.delete(self.algo_id)
            self.algo_id = None
        if self.vol_id is not None:
            adata.delete(self.vol_id)
            self.vol_id = None
        if self.sino_id is not None:
            adata.delete(self.sino_id)
            self.sino_id = None
        if self.proj_id is not None:
            aproj.delete(self.proj_id)
            self.proj_id = None


class AstraCudaBackProjectorImpl(object):
    """Thin wrapper around ASTRA."""

    algo_id = None
    vol_id = None
    sino_id = None
    proj_id = None

    def __init__(self, geometry, reco_space, proj_space):
        """Initialize a new instance.

        Parameters
        ----------
        geometry : `Geometry`
            Geometry defining the tomographic setup.
        reco_space : `DiscreteLp`
            Reconstruction space, the space to which the backprojection maps.
        proj_space : `DiscreteLp`
            Projection space, the space from which the backprojection maps.
        """
        assert isinstance(geometry, Geometry)
        assert isinstance(reco_space, DiscreteLp)
        assert isinstance(proj_space, DiscreteLp)

        self.geometry = geometry
        self.reco_space = reco_space
        self.proj_space = proj_space
        self.proj_ptr = None
        self.vol_ptr = None
        self.last_angles = None
        # self.create_ids(proj_tensor, volume_tensor)

        # Create a mutually exclusive lock so that two callers cant use the
        # same shared resource at the same time.
        self._mutex = Lock()

    def call_backward(self, proj_tensor, vol_tensor, angles=None):
        """Run an ASTRA back-projection on the given data using the GPU.

        Parameters
        ----------
        proj_data : ``proj_space`` element
            Projection data to which the back-projector is applied.
        vol_tensor : ``reco_space`` element, optional
            Element of the reconstruction space to which the result is written.
            If ``None``, an element in ``reco_space`` is created.

        Returns
        -------
        vol_tensor : ``reco_space`` element
            Reconstruction data resulting from the application of the
            back-projector. If ``vol_tensor`` was provided, the returned object is a
            reference to it.
        """
        with self._mutex:
            have_new_angles = self.last_angles is not None and self.last_angles != angles
            if self.proj_ptr is None or self.vol_ptr is None or self.proj_ptr != proj_tensor.data_ptr() or self.vol_ptr != vol_tensor.data_ptr() or have_new_angles:
                self.geometry.implementation_cache = {}
                self.create_ids2(proj_tensor, vol_tensor, angles)

            # Run algorithm
            astra.algorithm.run(self.algo_id)

            # Fix scaling to weight by pixel/voxel size
            vol_tensor *= astra_cuda_bp_scaling_factor(
                self.proj_space, self.reco_space, self.geometry)

            return vol_tensor

    def create_ids2(self, proj_tensor, vol_tensor, angles=None):
        """Create ASTRA objects."""
        self.proj_ptr = proj_tensor.data_ptr()
        self.vol_ptr = vol_tensor.data_ptr()
        if self.geometry.motion_partition.ndim == 1:
            motion_shape = self.geometry.motion_partition.shape
        else:
            # Need to flatten 2- or 3-dimensional angles into one axis
            motion_shape = (np.prod(self.geometry.motion_partition.shape),)

        proj_shape = motion_shape + self.geometry.det_partition.shape
        proj_ndim = len(proj_shape)

        if proj_ndim == 2:
            astra_proj_shape = proj_shape
            astra_vol_shape = self.reco_space.shape
        elif proj_ndim == 3:
            # The `u` and `v` axes of the projection data are swapped,
            # see explanation in `astra_*_3d_geom_to_vec`.
            astra_proj_shape = (proj_shape[1], proj_shape[0], proj_shape[2])
            astra_vol_shape = vol_tensor.shape

        # if proj_tensor.shape != astra_proj_shape:
        #     raise RuntimeError(
        #         f"proj_tensor.shape != astra_proj_shape : proj_tensor.shape = {proj_tensor.shape}, astra_proj_shape = {astra_proj_shape}")
        if vol_tensor.shape != astra_vol_shape:
            raise RuntimeError(
                f"vol_tensor.shape != astra_vol_shape : vol_tensor.shape = {vol_tensor.shape}, astra_vol_shape = {astra_vol_shape}")

        self.in_array = vol_tensor
        self.out_array = proj_tensor

        z, y, x = vol_tensor.shape
        vol_stride = th.tensor(vol_tensor.stride()) * vol_tensor.storage().element_size()
        vol_link = astra.data3d.GPULink(vol_tensor.storage().data_ptr(), x, y, z, vol_stride[-2])

        z, y, x = proj_tensor.shape
        proj_stride = th.tensor(proj_tensor.stride()) * proj_tensor.storage().element_size()
        proj_data_link = astra.data3d.GPULink(proj_tensor.storage().data_ptr(), x, y, z, proj_stride[-2])

        # Create ASTRA data structures
        vol_geom = astra_volume_geometry(self.reco_space)
        proj_geom = astra_projection_geometry2(self.geometry, angles)
        self.vol_id = astra.data3d.link('-vol', vol_geom, vol_link)
        self.sino_id = astra.data3d.link('-sino', proj_geom, proj_data_link)

        self.proj_id = astra_projector('nearest', vol_geom, proj_geom,
                                       ndim=proj_ndim, impl='cuda')

        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['ReconstructionDataId'] = self.vol_id
        cfg['ProjectionDataId'] = self.sino_id
        self.algo_id = astra.algorithm.create(cfg)

        # Create algorithm
        # self.algo_id = astra_algorithm(
        #     'backward', proj_ndim, self.vol_id, self.sino_id,
        #     proj_id=self.proj_id, impl='cuda')

    def create_ids(self, proj_tensor, vol_tensor):
        """Create ASTRA objects."""
        self.proj_ptr = proj_tensor.data_ptr()
        self.vol_ptr = vol_tensor.data_ptr()
        if self.geometry.motion_partition.ndim == 1:
            motion_shape = self.geometry.motion_partition.shape
        else:
            # Need to flatten 2- or 3-dimensional angles into one axis
            motion_shape = (np.prod(self.geometry.motion_partition.shape),)

        proj_shape = motion_shape + self.geometry.det_partition.shape
        proj_ndim = len(proj_shape)

        if proj_ndim == 2:
            astra_proj_shape = proj_shape
            astra_vol_shape = self.reco_space.shape
        elif proj_ndim == 3:
            # The `u` and `v` axes of the projection data are swapped,
            # see explanation in `astra_*_3d_geom_to_vec`.
            astra_proj_shape = (proj_shape[1], proj_shape[0], proj_shape[2])
            astra_vol_shape = vol_tensor.shape

        if proj_tensor.shape != astra_proj_shape:
            raise RuntimeError(
                f"proj_tensor.shape != astra_proj_shape : proj_tensor.shape = {proj_tensor.shape}, astra_proj_shape = {astra_proj_shape}")
        if vol_tensor.shape != astra_vol_shape:
            raise RuntimeError(
                f"vol_tensor.shape != astra_vol_shape : vol_tensor.shape = {vol_tensor.shape}, astra_vol_shape = {astra_vol_shape}")

        self.in_array = vol_tensor
        self.out_array = proj_tensor

        z, y, x = vol_tensor.shape
        vol_stride = th.tensor(vol_tensor.stride()) * vol_tensor.storage().element_size()
        vol_link = astra.data3d.GPULink(vol_tensor.storage().data_ptr(), x, y, z, vol_stride[-2])

        z, y, x = proj_tensor.shape
        proj_stride = th.tensor(proj_tensor.stride()) * proj_tensor.storage().element_size()
        proj_data_link = astra.data3d.GPULink(proj_tensor.storage().data_ptr(), x, y, z, proj_stride[-2])

        # Create ASTRA data structures
        vol_geom = astra_volume_geometry(self.reco_space)
        proj_geom = astra_projection_geometry(self.geometry)
        self.vol_id = astra.data3d.link('-vol', vol_geom, vol_link)
        self.sino_id = astra.data3d.link('-sino', proj_geom, proj_data_link)

        self.proj_id = astra_projector('nearest', vol_geom, proj_geom,
                                       ndim=proj_ndim, impl='cuda')

        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['ReconstructionDataId'] = self.vol_id
        cfg['ProjectionDataId'] = self.sino_id
        self.algo_id = astra.algorithm.create(cfg)

        # Create algorithm
        # self.algo_id = astra_algorithm(
        #     'backward', proj_ndim, self.vol_id, self.sino_id,
        #     proj_id=self.proj_id, impl='cuda')

    def __del__(self):
        """Delete ASTRA objects."""
        if self.geometry.ndim == 2:
            adata, aproj = astra.data2d, astra.projector
        else:
            adata, aproj = astra.data3d, astra.projector3d

        if self.algo_id is not None:
            astra.algorithm.delete(self.algo_id)
            self.algo_id = None
        if self.vol_id is not None:
            adata.delete(self.vol_id)
            self.vol_id = None
        if self.sino_id is not None:
            adata.delete(self.sino_id)
            self.sino_id = None
        if self.proj_id is not None:
            aproj.delete(self.proj_id)
            self.proj_id = None


def astra_cuda_bp_scaling_factor(proj_space, reco_space, geometry):
    """Volume scaling accounting for differing adjoint definitions.

    ASTRA defines the adjoint operator in terms of a fully discrete
    setting (transposed "projection matrix") without any relation to
    physical dimensions, which makes a re-scaling necessary to
    translate it to spaces with physical dimensions.

    Behavior of ASTRA changes slightly between versions, so we keep
    track of it and adapt the scaling accordingly.
    """
    # Angular integration weighting factor
    # angle interval weight by approximate cell volume
    angle_extent = geometry.motion_partition.extent
    num_angles = geometry.motion_partition.shape
    # TODO: this gives the wrong factor for Parallel3dEulerGeometry with
    # 2 angles
    scaling_factor = (angle_extent / num_angles).prod()

    # Correct in case of non-weighted spaces
    proj_extent = float(proj_space.partition.extent.prod())
    proj_size = float(proj_space.partition.size)
    proj_weighting = proj_extent / proj_size

    scaling_factor *= (proj_space.weighting.const /
                       proj_weighting)
    scaling_factor /= (reco_space.weighting.const /
                       reco_space.cell_volume)

    if parse_version(ASTRA_VERSION) < parse_version('1.8rc1'):
        if isinstance(geometry, Parallel2dGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(reco_space.cell_volume)
        elif isinstance(geometry, Parallel3dAxisGeometry):
            # Scales with voxel stride
            # In 1.7, only cubic voxels are supported
            voxel_stride = reco_space.cell_sides[0]
            scaling_factor /= float(voxel_stride)
        elif isinstance(geometry, ConeFlatGeometry):
            # Scales with 1 / cell_volume
            # In 1.7, only cubic voxels are supported
            voxel_stride = reco_space.cell_sides[0]
            scaling_factor /= float(voxel_stride)
            # Magnification correction
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius) ** 2

    else:
        if isinstance(geometry, Parallel2dGeometry):
            # Scales with 1 / cell_volume
            scaling_factor *= float(reco_space.cell_volume)
        elif isinstance(geometry, Parallel3dAxisGeometry):
            # Scales with cell volume
            # currently only square voxels are supported
            scaling_factor /= reco_space.cell_volume
        elif isinstance(geometry, ConeFlatGeometry):
            # Scales with cell volume
            scaling_factor /= reco_space.cell_volume
            # Magnification correction (scaling = 1 / magnification ** 2)
            src_radius = geometry.src_radius
            det_radius = geometry.det_radius
            scaling_factor *= ((src_radius + det_radius) / src_radius) ** 2

            # Correction for scaled 1/r^2 factor in ASTRA's density weighting.
            # This compensates for scaled voxels and pixels, as well as a
            # missing factor src_radius ** 2 in the ASTRA BP with
            # density weighting.
            det_px_area = geometry.det_partition.cell_volume
            scaling_factor *= (src_radius ** 2 * det_px_area ** 2 /
                               reco_space.cell_volume ** 2)

        # TODO: add case with new ASTRA release

    return scaling_factor


if __name__ == '__main__':
    from odl.util.testutils import run_doctests

    run_doctests()
