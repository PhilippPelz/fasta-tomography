# Copyright 2014-2017 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Tomography related operators and geometries."""
from __future__ import absolute_import
from .geometry import *
from .operators import *
from .util import *

def ray_transforms(real_space_extent, projection_shape, num_projections: int, interp='nearest'):
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
    reco_space = odl.uniform_discr(
        min_pt=[-real_space_extent[0] / 2, -real_space_extent[1] / 2, -real_space_extent[2] / 2],
        max_pt=[real_space_extent[0] / 2, real_space_extent[1] / 2, real_space_extent[2] / 2],
        shape=[projection_shape[0], projection_shape[1], projection_shape[1]],
        dtype='float32', interp=interp)
    angle_partition_dummy = odl.uniform_partition(
        min_pt=[-real_space_extent[0] / 2, -real_space_extent[1] / 2, -real_space_extent[2] / 2],
        max_pt=[real_space_extent[0] / 2, real_space_extent[1] / 2, real_space_extent[2] / 2],
        shape=[num_projections, projection_shape[0], projection_shape[1]])
    phi = np.linspace(0, np.deg2rad(90), int(np.ceil(num_projections ** (1 / 3))))
    theta = np.linspace(0, np.deg2rad(0.5), int(np.ceil(num_projections ** (1 / 3))))
    psi = np.linspace(0, np.deg2rad(0.5), int(np.ceil(num_projections ** (1 / 3))))
    angle_partition = odl.nonuniform_partition(phi, theta, psi)
    print('angle_partition',angle_partition)
    detector_partition = odl.uniform_partition([-real_space_extent[0] / 2, -real_space_extent[1] / 2],
                                               [real_space_extent[0] / 2, real_space_extent[1] / 2],
                                               [projection_shape[0], projection_shape[1]])
    print('detector_partition', detector_partition)
    geometry = Parallel3dEulerGeometry(angle_partition, detector_partition, check_bounds=False)
    print('geometry', geometry)
    print('angle_partition_dummy',angle_partition_dummy)
    range = odl.uniform_discr_frompartition(angle_partition_dummy, dtype=np.float32)
    ray_trafo = RayTransform(reco_space, geometry, impl='astra_cuda')
    rayback_trafo = RayBackProjection(range, geometry, impl='astra_cuda', domain=reco_space)
    return ray_trafo, rayback_trafo


__all__ = ()

from .geometry import *

__all__ += geometry.__all__

from .backends import *

__all__ += backends.__all__

from .operators import *

__all__ += operators.__all__

from .analytic import *

__all__ += analytic.__all__
