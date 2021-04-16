from fastatomography.util.parameters import Param
import torch as th
from tqdm import trange
from math import *
import numpy as np
from fastatomography.default_options import set_default_opts

def fasta(A, At, f, gradf, g, proxg, x0, opts):
    """
    :param A: A matrix (or optionally a function handle to a method) that
             returns A*x
    :param At: The adjoint (transpose) of 'A.' Optionally, a function handle
             may be passed.
    :param f: A function of x, computes the value of f
    :param gradf: A function of z, computes the gradient of f at z
    :param g: A function of x, computes the value of g
    :param proxg: A function of z and t, the proximal operator of g with
             stepsize t.
    :param x0: The initial guess, usually a vector of zeros
    :param opts: An optional struct with options.  The commonly used fields
             of 'opts' are:
               maxIters : (integer, default=1e4) The maximum number of iterations
                               allowed before termination.
               tol      : (double, default=1e-3) The stopping tolerance.
                               A smaller value of 'tol' results in more
                               iterations.
               verbose  : (boolean, default=false)  If true, print out
                               convergence information on each iteration.
               recordObjective:  (boolean, default=false) Compute and
                               record the objective of each iterate.
               recordIterates :  (boolean, default=false) Record every
                               iterate in a cell array.
            To use these options, set the corresponding field in 'opts'.
            For example:
                      >> opts.tol=1e-8;
                      >> opts.maxIters = 100;
    :return: a tuple (solution, out_dictionary, in_options)
    """
    opts = set_default_opts(opts, A, At, x0, gradf)

    tau1 = opts.tau
    max_iters = opts.max_iters
    W = opts.window

    residual = th.zeros(max_iters)
    normalized_resid = th.zeros(max_iters)
    taus = th.zeros(max_iters)
    fvals = th.zeros(max_iters)
    objective = th.zeros(max_iters + 1)
    func_values = th.zeros(max_iters)
    total_backtracks = 0
    backtrack_count = 0
    iterates = {}

    x1 = x0
    d1 = A(x1)
    f1 = f(d1)
    fvals[0] = f1
    gradf1 = At(gradf(d1))

    if opts.accelerate:
        x_accel1 = x0
        d_accel1 = d1
        alpha1 = 1

    max_residual = - np.inf
    min_objective_value = np.inf

    if opts.record_objective:
        objective[0] = f1 + g(x0)

    for i in trange(max_iters):
        x0 = x1
        gradf0 = gradf1
        tau0 = tau1

        x1hat = x0 - tau0 * gradf0
        x1 = proxg(x1hat, tau0)

        Dx = x1 - x0
        d1 = A(x1)
        f1 = f(d1)

        if opts.backtrack and i > 0:
            M = th.max(fvals[max(i - W, 0):max(i, 0)])
            backtrack_count = 0
            while f1 - 1e-12 > M + th.dot(Dx.view(-1), gradf0.view(-1)) + th.norm(Dx.view(-1)) ** 2 / (
                    2 * tau0) and backtrack_count < 20:
                tau0 = tau0 * opts.stepsize_shrink
                x1hat = x0 - tau0 * gradf0
                x1 = proxg(x1hat, tau0)
                d1 = A(x1)
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
            objective[i + 1] = f1 + g(x1)
            new_objective_value = objective[i + 1]
        else:
            new_objective_value = residual[i]

        if opts.record_iterates:
            iterates[f'{i}'] = x1

        if new_objective_value < min_objective_value:
            best_objective_iterate = x1
            min_objective_value = new_objective_value

        if opts.verbose:
            print()

        if opts.stop_now(x1, i, residual[i], normalized_resid[i], max_residual, opts) or i > max_iters:
            outs = Param()
            outs.solve_time = 0
            outs.residuals = residual[:i]
            outs.stepsizes = taus[1:i]
            outs.normalized_residuals = normalized_resid[1:i]
            outs.objective = objective[1:i]
            outs.func_values = func_values[1:i]
            outs.backtracks = total_backtracks
            outs.L = opts.L
            outs.initial_stepsize = opts.tau
            outs.iteration_count = i
            if opts.record_iterates:
                outs.iterates = iterates
            sol = best_objective_iterate
            return sol, outs, opts

        if opts.adaptive and ~opts.accelerate:
            gradf1 = At(gradf(d1))
            Dg = gradf1 + (x1hat - x0) / tau0
            dotprod = th.dot(Dx, Dg)
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
            gradf1 = At(gradf(d1))
            func_values[i] = f(d1)
            tau1 = tau0

        if ~opts.adaptive and ~opts.accelerate:
            gradf1 = At(gradf(d1))
            tau1 = tau0



def checkAdjoint(A, At, x):
    x = th.randn(x.shape)
    Ax = A(x)
    y = th.randn(Ax.shape)
    Aty = At(y)
    Ax.view((th.prod(th.tensor(x.shape)), 1)).t()
    innerProduct1 = Ax.view((th.prod(th.tensor(Ax.shape)), 1)).t() @ y.view(-1)
    innerProduct2 = x.view((th.prod(th.tensor(x.shape)), 1)).t() @ Aty.view(-1)
    error = abs(innerProduct1 - innerProduct2) / max(abs(innerProduct1), abs(innerProduct2))
    assert error < 1e-9, '"At" is not the adjoint of "A".  Check the definitions of these operators.'
    return


