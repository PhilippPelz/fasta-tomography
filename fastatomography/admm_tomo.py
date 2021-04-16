from fastatomography.default_dependencies import *
from fastatomography.tomo import ray_transforms

def prox_l2_data(x, data, sigma, Lambda):
    num = x + 2 * sigma * Lambda * data
    denom = 1 + 1 * sigma * Lambda
    return num/denom

def admm(it, projections, angles, real_space_extent, dtype=th.float32, solution=None):
    """
    .. math::
        x^{(k+1)} &= \mathrm{prox}_{\tau f} \left[
            x^{(k)} - \sigma^{-1}\tau L^*\big(
                L x^{(k)} - z^{(k)} + u^{(k)}
            \big)
        \right]

        z^{(k+1)} &= \mathrm{prox}_{\sigma g}\left(
            L x^{(k+1)} + u^{(k)}
        \right)

        u^{(k+1)} &= u^{(k)} + L x^{(k+1)} - z^{(k+1)}

    :param it: number of iterations
    :param projections: n_angles x ny x nx  measured projections
    :param angles: 3 x n_angles angles of measured projections
    :return: volume
    """
    sigma = 1
    tau = 0.5
    Lambda = 1
    projection_shape = projections.shape[1:]
    vol_shape = (projection_shape[0], projection_shape[1], projection_shape[1])

    tmp_dom = th.zeros(vol_shape, dtype=dtype)
    tmp_range = th.zeros(projections.shape, dtype=dtype)
    x = th.zeros(vol_shape, dtype=dtype)
    z_hat = th.zeros(projections.shape, dtype=dtype)
    u = th.zeros(projections.shape, dtype=dtype)

    A, AH = ray_transforms(real_space_extent, projections.shape[1:], angles.shape[1])

    for i in trange(it):
        z_hat += u
        z_hat -= z

        tmp_dom = AH(z_hat, out=tmp_dom, angles=angles)

        x = x - (tau / sigma) * tmp_dom

        z_hat = A(x, out=z_hat, angles=angles)

        z = prox_l2_data(z_hat + u, projections, sigma, Lambda)

        u += z_hat
        u -= z

        err = th.norm(th.abs(A(x, out=tmp_range, angles=angles)) - projections)/th.norm(projections)
        if solution is not None:
            err_sol = th.norm(x-solution)/th.norm(solution)
        else:
            err_sol = 0
        print(f"{i} measure_error: {err:3.3g} sol_error: {err_sol:3.3g}")

    return x