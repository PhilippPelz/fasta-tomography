from fastatomography.util.parameters import Param
import torch as th
from tqdm import trange
from math import *
import numpy as np
from fastatomography.default_options import set_default_opts_tomography
from fastatomography.util import R_factor

def fasta(A, At, f, gradf, g, proxg, x0, y, angles, opts, vol_grad_weights = None):
    opts = set_default_opts_tomography(opts, A, At, x0, y, angles, gradf)

    tau1 = opts.tau
    max_iters = opts.max_iters
    W = opts.window

    residual = th.zeros(max_iters)
    normalized_resid = th.zeros(max_iters)
    taus = th.zeros(max_iters)
    fvals = th.zeros(max_iters)
    objective = th.zeros(max_iters + 1)
    R_factors = th.zeros(max_iters + 1)
    func_values = th.zeros(max_iters)
    g_values = th.zeros(max_iters +1)
    total_backtracks = 0
    backtrack_count = 0
    iterates = {}

    x1 = x0

    d1 = th.zeros_like(y, device=x1.device, dtype=x1.dtype)
    gradf1 = th.zeros_like(x1, device=x1.device, dtype=x1.dtype)

    d1 = A(x1, out=d1, angles=angles)
    f1 = f(d1)
    fvals[0] = f1
    gradf1 = At(gradf(d1), out=gradf1, angles=angles)

    if opts.accelerate:
        x_accel1 = x0
        d_accel1 = d1
        alpha1 = 1

    max_residual = - np.inf
    min_objective_value = np.inf

    if opts.record_objective:
        g_values[0] = g(x0)
        objective[0] = f1 + g(x0)
        R_factors[0] = R_factor(d1, y)

    best_objective_iterate = None

    if opts.verbose:
        range_func = trange
    else:
        range_func = range

    for i in range_func(max_iters):
        x0 = x1
        gradf0 = gradf1
        tau0 = tau1

        if vol_grad_weights is None:
            x1hat = x0 - tau0 * gradf0
        else:
            x1hat = x0 - tau0 * gradf0 * vol_grad_weights
        x1 = proxg(x1hat, tau0)

        Dx = x1 - x0
        d1 = A(x1, out=d1, angles=angles)
        f1 = f(d1)

        if opts.backtrack and i > 0:
            M = th.max(fvals[max(i - W, 0):max(i, 0)])
            backtrack_count = 0
            while f1 - 1e-12 > M + th.dot(Dx.view(-1), gradf0.view(-1)) + th.norm(Dx.view(-1)) ** 2 / (
                    2 * tau0) and backtrack_count < 20:
                tau0 = tau0 * opts.stepsize_shrink
                if vol_grad_weights is None:
                    x1hat = x0 - tau0 * gradf0
                else:
                    x1hat = x0 - tau0 * gradf0 * vol_grad_weights
                x1 = proxg(x1hat, tau0)
                d1 = A(x1, out=d1, angles=angles)
                f1 = f(d1)
                Dx = x1 - x0
                backtrack_count += 1
            total_backtracks += backtrack_count
        if opts.verbose and backtrack_count > 10:
            print(f'WARNING: excessive backtracking ({backtrack_count} steps, current stepsize is {tau0}')

        taus[i] = tau0
        residual[i] = th.norm(Dx) / tau0
        max_residual = max(residual[i], max_residual)
        normalizer = max(th.norm(gradf0), th.norm(x1 - x1hat / tau0) + opts.eps_n)
        normalized_resid[i] = residual[i] / normalizer
        fvals[i] = f1
        func_values[i] = opts.function(x0)
        if opts.record_objective:
            g_values[i+1] = g(x1)
            objective[i + 1] = f1 + g_values[i+1]
            new_objective_value = objective[i + 1]
            R_factors[i + 1] = R_factor(d1, y)
        else:
            new_objective_value = residual[i]

        if opts.record_iterates:
            iterates[f'{i}'] = x1

        if new_objective_value < min_objective_value:
            best_objective_iterate = x1
            min_objective_value = new_objective_value

        if opts.verbose:
            # print()
            pass

        if opts.stop_now(x1, i, residual[i], normalized_resid[i], max_residual, opts) or i == max_iters - 1:
            outs = Param()
            outs.solve_time = 0
            outs.residuals = residual[:i]
            outs.stepsizes = taus[:i]
            outs.normalized_residuals = normalized_resid[:i]
            outs.objective = objective[:i+1]
            outs.g_values = g_values[:i+1]
            outs.func_values = func_values[:i+1]
            outs.backtracks = total_backtracks
            outs.R_factors = R_factors[:i+1]
            outs.L = opts.L
            outs.initial_stepsize = opts.tau
            outs.iteration_count = i
            if opts.record_iterates:
                outs.iterates = iterates
            if best_objective_iterate is not None:
                sol = best_objective_iterate
            else:
                sol = x1
            return sol, outs, opts

        if opts.adaptive and ~opts.accelerate:
            gradf1 = At(gradf(d1), out=gradf1, angles=angles)
            Dg = gradf1 + (x1hat - x0) / tau0
            dotprod = th.dot(Dx.view(-1), Dg.view(-1))
            tau_s = th.norm(Dx) ** 2 / dotprod
            tau_m = dotprod / th.norm(Dg) ** 2
            tau_m = max(tau_m, 0)
            if 2 * tau_m > tau_s:
                tau1 = tau_m
            else:
                tau1 = tau_s - 0.5 * tau_m
            if tau1 <= 0 or th.isinf(tau1) or th.isnan(tau1):
                tau1 = tau0 * 1.5

        if opts.accelerate:
            x_accel0 = x_accel1
            d_accel0 = d_accel1
            alpha0 = alpha1
            x_accel1 = x1
            d_accel1 = d1
            if opts.restart and th.dot((x0 - x1).view(-1), (x_accel1 - x_accel0).view(-1)) > 0:
                alpha0 = 1
            alpha1 = (1 + sqrt(1 + 4 * alpha0 ** 2)) / 2
            x1 = x_accel1 + (alpha0 - 1) / alpha1 * (x_accel1 - x_accel0)
            d1 = d_accel1 + (alpha0 - 1) / alpha1 * (d_accel1 - d_accel0)
            # Compute the gradient needed on the next iteration
            gradf1 = At(gradf(d1), out=gradf1, angles=angles)
            func_values[i] = f(d1)
            tau1 = tau0

        if ~opts.adaptive and ~opts.accelerate:
            gradf1 = At(gradf(d1), out=gradf1, angles=angles)
            tau1 = tau0


def checkAdjoint(A, At, x):
    x = th.randn(x.shape)
    Ax = A(x)
    y = th.randn(Ax.shape)
    Aty = At(y)
    innerProduct1 = Ax.view((th.prod(th.tensor(Ax.shape)), 1)).t() @ y.view(-1)
    innerProduct2 = x.view((th.prod(th.tensor(x.shape)), 1)).t() @ Aty.view(-1)
    error = abs(innerProduct1 - innerProduct2) / max(abs(innerProduct1), abs(innerProduct2))
    assert error < 1e-9, '"At" is not the adjoint of "A".  Check the definitions of these operators.'
    return
