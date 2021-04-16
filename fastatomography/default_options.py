import torch as th


def set_defaults_opts_base(opts, A, At, x0, gradf):
    valid_options = ['max_iters', 'tol', 'verbose', 'record_objective',
                     'record_iterates', 'adaptive', 'accelerate', 'restart', 'backtrack',
                     'stepsize_shrink', 'window', 'eps_r', 'eps_n', 'L', 'tau', 'function',
                     'string_header', 'stop_rule', 'stop_now', 'mode']
    for k in opts.keys():
        if not k in valid_options:
            raise RuntimeError(
                f'invalid option supplied to fasta: {k}.   Valid choices are:  max_iters, tol, verbose, record_objective, record_iterates,  adaptive, accelerate, restart, backtrack, stepsize_shrink,  window, eps_r, eps_n, L, tau, function, string_header,  stop_rule, stop_now.')

    if not 'max_iters' in opts:
        opts.max_iters = 1000
    if not 'tol' in opts:
        opts.tol = 1e-3
    if not 'verbose' in opts:
        opts.verbose = False
    if not 'record_objective' in opts:
        opts.record_objective = False
    if not 'record_iterates' in opts:
        opts.record_iterates = False
    if not 'adaptive' in opts:
        opts.adaptive = True
    if not 'accelerate' in opts:
        opts.accelerate = False
    if not 'restart' in opts:
        opts.restart = True
    if not 'backtrack' in opts:
        opts.backtrack = True
    if not 'stepsize_shrink' in opts:
        opts.stepsize_shrink = 0.2
        if ~opts.adaptive or opts.accelerate:
            opts.stepsize_shrink = 0.5
    if not 'window' in opts:
        opts.window = 10
    if not 'eps_r' in opts:
        opts.eps_r = 1e-8
    if not 'eps_n' in opts:
        opts.eps_n = 1e-8
    opts.mode = 'plain'
    if opts.adaptive:
        opts.mode = 'adaptive'
    if opts.accelerate:
        if opts.restart:
            opts.mode = 'accelerated(FISTA)+restart'
        else:
            opts.mode = 'accelerated(FISTA)'
    if not 'function' in opts:
        opts.function = lambda x: 0
    if not 'string_header' in opts:
        opts.string_header = ''
    if not 'stop_now' in opts:
        opts.stop_rule = 'hybridResidual'
    if opts.stop_rule == 'hybridResidual':
        def residual(x1, iter, resid, normResid, maxResidual, opts):
            return resid < opts.tol

        opts.stop_now = residual
    if opts.stop_rule == 'iterations':
        def iterations(x1, iter, resid, normResid, maxResidual, opts):
            return iter > opts.maxIters

        opts.stop_now = iterations
    if opts.stop_rule == 'normalizedResidual':
        def normalizedResidual(x1, iter, resid, normResid, maxResidual, opts):
            return normResid < opts.tol

        opts.stop_now = normalizedResidual
    if opts.stop_rule == 'ratioResidual':
        def ratioResidual(x1, iter, resid, normResid, maxResidual, opts):
            return resid / (maxResidual + opts.eps_r) < opts.tol

        opts.stop_now = ratioResidual
    if opts.stop_rule == 'hybridResidual':
        def hybridResidual(x1, iter, resid, normResid, maxResidual, opts):
            return (resid / (maxResidual + opts.eps_r) < opts.tol) or normResid < opts.tol

        opts.stop_now = hybridResidual

    assert 'stop_now' in opts, f'Invalid choice of stopping rule: {opts.stop_rule}'

    return opts


def set_default_opts_tomography(opts, A, At, x0, y0, angles, gradf):
    opts = set_defaults_opts_base(opts, A, At, x0, gradf)
    if (not 'L' in opts or opts.L <= 0) and (not 'tau' in opts or opts.tau <= 0):
        x1 = th.randn_like(x0)
        x2 = th.randn_like(x0)
        y1 = th.randn_like(y0)
        y2 = th.randn_like(y0)
        gradf1 = th.zeros_like(x0)
        gradf2 = th.zeros_like(x0)
        gradf1 = At(gradf(A(x1, out=y1, angles=angles)), out=gradf1, angles=angles)
        gradf2 = At(gradf(A(x2, out=y2, angles=angles)), out=gradf2, angles=angles)
        opts.L = th.norm(gradf1 - gradf2) / th.norm(x2 - x1)
        opts.L = max(opts.L, 1e-6)
        opts.tau = 2 / opts.L / 10
    assert opts.tau > 0, f'Invalid step size: {opts.tau}'
    if not 'tau' in opts or opts.tau <= 0:
        opts.tau = 1.0 / opts.L
    else:
        opts.L = 1 / opts.tau
    return opts


def set_default_opts(opts, A, At, x0, gradf):
    opts = set_defaults_opts_base(opts, A, At, x0, gradf)
    if (not 'L' in opts or opts.L <= 0) and (not 'tau' in opts or opts.tau <= 0):
        x1 = th.randn(x0.shape)
        x2 = th.randn(x0.shape)
        gradf1 = At(gradf(A(x1)))
        gradf2 = At(gradf(A(x2)))
        opts.L = th.norm(gradf1 - gradf2) / th.norm(x2 - x1)
        opts.L = max(opts.L, 1e-6)
        opts.tau = 2 / opts.L / 10
    assert opts.tau > 0, f'Invalid step size: {opts.tau}'
    if not 'tau' in opts or opts.tau <= 0:
        opts.tau = 1.0 / opts.L
    else:
        opts.L = 1 / opts.tau
    return opts
