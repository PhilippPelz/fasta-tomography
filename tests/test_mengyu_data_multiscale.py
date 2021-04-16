import matplotlib as mpl
from numpy.random import uniform
from scipy.ndimage import fourier_shift
from skimage.feature import register_translation
from tqdm import trange
# mpl.rcParams['text.usetex'] = True
from fastatomography.fasta_tomography import fasta_tomography_nonnegative_shrink
from fastatomography.default_dependencies import *
from scipy.io import loadmat
from skimage.filters import gaussian
from fastatomography.util import *
from fastatomography.tomo import ray_transforms
from fastatomography.util.plotting import plot_rotating_point, plot_translations, save_stack_movie, save_stack_gif

# load data
path = '/home/philipp/projects2/tomo/2020-04-20_mengyu_nanowire/'
fig_path = '/home/philipp/projects2/tomo/2020-04-20_mengyu_nanowire/'
fn = 'stack.mat'
angles_fn = 'angles.mat'
dx = 1

proj = loadmat(path + fn)['stack']
ang = loadmat(path + angles_fn)['angles']
#%%
angles_in = np.deg2rad(ang)
angles_in = th.as_tensor(angles_in.T).squeeze().contiguous()

angles = angles_in.numpy()
n_angles = angles.shape[1]
# %%
pstack = np.transpose(proj, (2, 1, 0))
N = np.array(pstack.shape[1:])
n_angles = pstack.shape[0]
plotmosaic(pstack, 'Tilt series overview', dpi=600, savePath=f'{fig_path}series_overview')
# pad in real space to alleviate aliasing artifacts
pstack = np.pad(pstack, ((0, 0), (N[0] // 3, N[0] // 3), (N[0] // 3, N[0] // 3)), mode='constant', constant_values=0)
N = np.array(pstack.shape[1:])
#%%

#%%
# save_stack_movie(path+'gif', pstack, np.rad2deg(angles_in[1]), dx)
#%%
# save_stack_gif(path+'gif', pstack, np.rad2deg(angles_in[1]), dx)
# %% determine resolution levels and resolution cutoff
q = fourier_coordinates_2D(N, [dx, dx], centered=False)
qn = np.linalg.norm(q, axis=0)
dq = qn[0, 1]
fpstack = fft2(pstack, norm='ortho')
ps = np.var(fftshift(np.abs(fpstack) ** 2), 0)
avg_ps = np.log10(ps)

resolution_cutoff = 1
resolutions = np.array([resolution_cutoff])
upsample_factors = resolutions / resolution_cutoff
q1 = 1 / (2 * resolutions)
r = np.ceil(q1 / dq)

falloff = 20
qcutoff = 1 / (2 * resolution_cutoff)
rcutoff = qcutoff / dq
cutoff_resolution_mask = sector_mask(N, N // 2, rcutoff, (0, 360)).astype(np.float32)
cutoff_resolution_mask = gaussian(cutoff_resolution_mask, falloff)

fig, ax = plt.subplots(dpi=150)
im = ax.imshow(avg_ps * cutoff_resolution_mask, interpolation='nearest', cmap=plt.cm.get_cmap('inferno'))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_title('log10(var(abs(fft2(stack))**2))')
ax.grid(False)
for r1, res in zip(r, resolutions):
    circle1 = plt.Circle(N / 2, r1, color='b', fill=None)
    ax.add_artist(circle1)
    txt = ax.text(*(N // 2 - r1), f'{res:2.2f} A', color='b')
plt.show()
fig.savefig(f'{fig_path}log_var.png', dpi=300)
# %% create datasets at different resolutions
# pad to 2x full resolution, so that maximum resolved lattice spacing has 4 pixels instead of 2
# max_size = np.array([4 * r[-1], 4 * r[-1]], dtype=np.int)
# pad = (max_size - N) // 2
# pad[pad < 0] = 0
# padding = ((0, 0), (pad[0], pad[0]), (pad[1], pad[1]))
# fpstack_fullres = fftshift(np.pad(fftshift(fpstack, (1, 2)), padding, mode='constant', constant_values=0), (1, 2))
# N_fullres = np.array(fpstack_fullres.shape[1:])
# N_stack = []
# cutoff_resolution_mask = sector_mask(N_fullres, N_fullres // 2, rcutoff, (0, 360)).astype(np.float32)
# cutoff_resolution_mask = fftshift(gaussian(cutoff_resolution_mask, falloff))
# fpstack_fullres *= cutoff_resolution_mask
# resolution_ratios = upsample_factors * 2
# stacks = []
# for i, (r0, res_ratio) in enumerate(zip(r, resolution_ratios)):
#     N_res = np.array([4 * r0, 4 * r0]).astype(np.int)
#     resolution_mask = np.prod(
#         np.abs(fourier_coordinates_2D(N_fullres, 1 / N_fullres, centered=True)) < N_res[:, None, None] // 2,
#         axis=0).astype(np.bool)
#     s = fpstack_fullres[:, resolution_mask].reshape((n_angles, *N_res))
#     # if r0 == resolutions[-1]:
#     w = slepian_window(N_res, res_ratio)
#     # else:
#     #     w = slepian_window(N_res, 2)
#     s *= w
#     stacks.append(s)
#     N_stack.append(N_res)
#
#     ps1 = np.var(fftshift(np.abs(s) ** 2), 0)
#     avg_ps1 = np.log10(ps1)
#     avg_ps1[avg_ps1 < 0] = 0
#     plot(avg_ps1, f'Log variance for {res_ratio / 2}x upsampling',
#          savePath=f'{fig_path}log_var_upsampled{res_ratio / 2}x')
#
# res_stacks = [ifft2(s, norm='ortho').real for s in stacks]
#
# # normalize stacks to highest resolution
# t = np.sum(res_stacks[-1][n_angles // 2])
# for s in res_stacks[:-1]:
#     ts = np.sum(s[n_angles // 2])
#     s *= t / ts
#
# for i, (s, ups) in enumerate(zip(res_stacks, upsample_factors)):
#     print(f'stack {i} sum: {np.sum(s[n_angles // 2])}')
#     plotmosaic(s, f'Tilt series overview {ups}x downsampling', dpi=600,
#                savePath=f'{fig_path}tilt_series_upsampled{res_ratio / 2}x')
# %% do angle refinement on low-resolution data
opts = Param()
opts.record_objective = True
opts.verbose = True
opts.string_header = '     '
opts.max_iters = 200
opts.tol = 1e-5
opts.accelerate = True
opts.adaptive = True
opts.restart = True
#step size, is picked automatgically if you leave this out. Sometimes the convegence curve looks weird,
# and you have to set it manually
opts.tau = 1e2

#
refinement_iterations = 20
i_refine = 0
refinement_objectives = np.zeros((refinement_iterations,))
angle_trial_start_range = np.deg2rad(1)
angle_trials = 5

pstack1 = pstack * (pstack.max()/pstack.max(axis=(1,2)))[:,None,None]
m = 42
stack_res3 = pstack1[:, m:-m, m:-m]

i_res = 0
stack_res = stack_res3
# N_res = N_stack[i_res]
registration_upsample_factor = 50
resolution_upsample_factor = upsample_factors[i_res]
translation_shifts = np.zeros((n_angles, 2))

#
plot(np.sum(stack_res, 0), 'sum(stack res)')

ps = stack_res.shape
projection_shape = ps[1:]
vol_shape = (projection_shape[0], projection_shape[1], projection_shape[1])
real_space_extent = np.array([projection_shape[0], projection_shape[0], projection_shape[1]])

A, At = ray_transforms(real_space_extent, projection_shape, n_angles)
x0 = th.zeros(*vol_shape, dtype=th.float32).cuda()

# % check for best regularization parameter
# mus = np.array([1e2, 5e1, 1e1, 5e0, 1e0, 5e-1, 1e-1, 5e-2, 1e-1, 5e-3, 1e-3])
# mu_objectives = np.zeros_like(mus)
# mu_g_values = np.zeros_like(mus)
# mu_solution_norm = np.zeros_like(mus)
# for i, mu in enumerate(mus):
#     sol, out, opts = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, angles, opts)
#     mu_objectives[i] = out.objective[-1]
#     mu_g_values[i] = out.g_values[-1]
#     mu_solution_norm[i] = th.norm(sol)
# font = {'family': 'serif',
#         'color': 'black',
#         'weight': 'normal',
#         'size': 6,
#         }
# f, a = plt.subplots(1, 2, dpi=300)
# a[0].loglog(mu_g_values, mu_objectives)
# a[0].set_ylabel('regularizer loss')
# a[0].set_xlabel('objective loss')
# for g, o, mu in zip(mu_g_values, mu_objectives, mus):
#     a[0].text(g, o, f'mu = {mu:2.2g}', fontdict=font)
# a[1].loglog(mu_objectives, mu_solution_norm)
# a[1].set_ylabel('solution norm')
# a[1].set_xlabel('objective loss')
# for g, o, mu in zip(mu_objectives, mu_solution_norm, mus):
#     a[1].text(g, o, f'mu = {mu:2.2g}', fontdict=font)
# plt.show()
# now reconstruct again with best parameter

best_angles = angles
mu = 1e-6


x0 = th.zeros(*vol_shape, dtype=th.float32).cuda()
y = th.as_tensor(np.transpose(stack_res3, (1, 0, 2))).contiguous().float().cuda()
# y *= 10000
print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing ...')
sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts)
#%%
rec = sol.cpu().numpy()
f, a = plt.subplots(1, 3, dpi=300)
a[0].imshow(rec[vol_shape[0] // 2, :, :])
a[0].set_xlabel(f'z')
a[0].set_ylabel(f'x')
a[1].imshow(rec[:, vol_shape[0] // 2, :])
a[1].set_xlabel(f'z')
a[1].set_ylabel(f'y')
a[2].imshow(rec[:, :, vol_shape[0] // 2])
a[2].set_xlabel(f'x')
a[2].set_ylabel(f'y')
plt.show()

print(out.R_factors)
# %% refinement iteration
for i_refine in range(refinement_iterations):
    x0 = th.zeros(*vol_shape, dtype=th.float32).cuda()
    y = th.as_tensor(np.transpose(stack_res, (1, 0, 2))).contiguous().float().cuda()
    print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing ...')
    sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts)
    refinement_objectives[i_refine] = out.objective[-1]
    # register projection with subpixel precision
    print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: subpixel registration ...')
    y_model = th.zeros_like(y)
    y_model = A(sol, out=y_model, angles=best_angles)
    y_model = th.transpose(y_model, 0, 1).cpu().numpy()
    precision = registration_upsample_factor * resolution_upsample_factor
    rms_translation_errors = np.zeros((n_angles,), dtype=np.float32)
    current_shifts = np.zeros((n_angles, 2))
    for i in range(n_angles):
        shift, rms_translation_errors[i], diffphase = register_translation(y_model[i], stack_res[i], precision)
        # print(f'angle {i:-4d} shift: {shift}')
        translation_shifts[i] += shift
        current_shifts[i] = shift
        stack_res[i] = ifftn(fourier_shift(fftn(stack_res[i]), shift))
    print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: max. shift: {np.max(np.abs(current_shifts))}')
    # plot_translations(current_shifts, f'Translations iteration {i_refine}',
    #                   savePath=f'{fig_path}{i_refine:03d}_translations')
    # plotmosaic(np.abs(y_model - stack_res), f'Tilt series residual after translations iteration {i_refine}',
    #            dpi=600, savePath=f'{fig_path}{i_refine:03d}_tilt_series_residuals')

    print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing for angle refinement ...')
    sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts)
    # %
    # compute trial angle losses and find best
    angle_trial_range = (1 - (i_refine / refinement_iterations)) * angle_trial_start_range
    print(
        f'it {i_refine:-4d}/{refinement_iterations:-4d}: angle refinement with range {np.rad2deg(angle_trial_range)} deg ...')
    trial_losses = np.zeros((angle_trials + 1, n_angles))
    trial_angles = np.zeros((angle_trials + 1, *angles.shape))
    for t in range(angle_trials):
        random_offsets = uniform(-angle_trial_range, angle_trial_range, angles.shape)
        trial_angles[t] = random_offsets + best_angles
        trial_proj = th.zeros_like(y)
        trial_proj = A(sol, out=trial_proj, angles=trial_angles[t])
        trial_losses[t] = (th.norm(trial_proj - y, dim=(0, 2)) ** 2).cpu().numpy()
        trial_angles[-1] = best_angles
    trial_proj = th.zeros_like(y)
    trial_proj = A(sol, out=trial_proj, angles=trial_angles[t])
    trial_losses[-1] = (th.norm(trial_proj - y, dim=(0, 2)) ** 2).cpu().numpy()
    min_loss = np.min(trial_losses, axis=0)
    mini_ind = np.argmin(trial_losses, axis=0)
    print(f'min indices: {mini_ind}')

    # pick angles with minimum error
    best_angles = np.squeeze(
        np.stack([a[mini_ind[i]] for i, a in enumerate(np.split(trial_angles, np.arange(1, n_angles), axis=2))])).T

plot_rotating_point(best_angles, f'Best angles iteration {i_refine}', dpi=600, savePath=f'{fig_path}best_angles')
# # %%
# np.save(path + 'fasta.npy', sol.cpu().numpy())
# %%
rec = sol.cpu().numpy()
f, a = plt.subplots(1, 3, dpi=300)
a[0].imshow(rec[vol_shape[0] // 2, :, :])
a[0].set_xlabel(f'z')
a[0].set_ylabel(f'x')
a[1].imshow(rec[:, vol_shape[0] // 2, :])
a[1].set_xlabel(f'z')
a[1].set_ylabel(f'y')
a[2].imshow(rec[:, :, vol_shape[0] // 2])
a[2].set_xlabel(f'x')
a[2].set_ylabel(f'y')
plt.show()
f.savefig(f'{fig_path}xyz_view.png', dpi=600)
# %%

f, a = plt.subplots(1, 3, dpi=300)
a[0].semilogy(out.residuals)
a[0].set_title(r"$||x_n-x_{n+1}||^2$")
a[1].semilogy(out.objective)
a[1].set_title(r"$||y-Ax||^2$")
a[2].semilogy(out.R_factors)
a[2].set_title(r"R-factor $=\frac{||y-Ax||_1}{||y||_1}$")
plt.show()
fig.savefig(f'{fig_path}residuals_objective.png', dpi=300)
# %%
plot_rotating_point(best_angles, f'Best angles iteration {i_refine}', savePath=f'{fig_path}best_angles.png', dpi=300)
plot_rotating_point(angles, f'Start angles iteration {i_refine}', savePath=f'{fig_path}start_angles.png', dpi=300)
# %%
# stack_res2 = res_stacks[-1].copy()
# for i in range(n_angles):
#     stack_res2[i] = ifftn(fourier_shift(fftn(stack_res2[i]), translation_shifts[i] * upsample_factors[0]))
# # %%
# plotmosaic(stack_res2, dpi=600, savePath=f'{fig_path}stack_aligned_fullres.png')
# # %%
# m = 160
# stack_res3 = stack_res2[:, m:-m, m:-m]
#
# ps = stack_res3.shape
# projection_shape = ps[1:]
# vol_shape = (projection_shape[0], projection_shape[1], projection_shape[1])
# real_space_extent = np.array([projection_shape[0], projection_shape[0], projection_shape[1]])
#
# opts = Param()
# opts.record_objective = True
# opts.verbose = True
# opts.string_header = '     '
# opts.max_iters = 450
# opts.tol = 1e-5
# opts.accelerate = True
# opts.adaptive = True
# opts.restart = True
# opts.tau = 1e-1
# mu = 5e-2
#
# A, At = ray_transforms(real_space_extent, projection_shape, n_angles)
# x0 = th.zeros(*vol_shape, dtype=th.float32).cuda()
# y = th.as_tensor(np.transpose(stack_res3, (1, 0, 2))).contiguous().float().cuda()
# print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing ...')
# sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts)
# # %%
# rec = sol.cpu().numpy()
# f, a = plt.subplots(1, 3, dpi=300)
# a[0].imshow(rec[vol_shape[0] // 2, :, :])
# a[0].set_xlabel(f'z')
# a[0].set_ylabel(f'x')
# a[1].imshow(rec[:, vol_shape[0] // 2, :])
# a[1].set_xlabel(f'z')
# a[1].set_ylabel(f'y')
# a[2].imshow(rec[:, :, vol_shape[0] // 2])
# a[2].set_xlabel(f'x')
# a[2].set_ylabel(f'y')
# plt.show()
# f.savefig(f'{fig_path}xyz_fullres.png', dpi=300)
# # %%
# f, a = plt.subplots(1, 3, dpi=300)
# a[0].semilogy(out.residuals)
# a[0].set_title(r"$||x_n-x_{n+1}||^2$")
# a[1].semilogy(out.objective)
# a[1].set_title(r"$||y-Ax||^2$")
# a[2].semilogy(out.R_factors)
# a[2].set_title(r"R-factor $=\frac{||y-Ax||_1}{||y||_1}$")
# plt.show()
# f.savefig(f'{fig_path}residuals_objective__fullres.png', dpi=300)
# print(f'best R-factor: {out.R_factors[-1]}')
# # %%
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# #
# # path1 = '/home/philipp/drop/Public/'
# # title = ['x-z', 'y-z', 'y-x']
# # cmap = 'viridis'
# # scale = (40 / dx, '4 nm')
# #
# # im1, im2, im3 = [rec[0, :, :], rec[:, 0, :], rec[:, :, 0]]
# # fig = plt.figure(dpi=600)
# # gs1 = gridspec.GridSpec(1, 3)
# # gs1.update(wspace=0, hspace=0)  # set the spacing between axes.
# # ax1 = plt.subplot(gs1[0, 0])
# # ax2 = plt.subplot(gs1[0, 1])
# # ax3 = plt.subplot(gs1[0, 2])
# # div1 = make_axes_locatable(ax1)
# # div2 = make_axes_locatable(ax2)
# # div3 = make_axes_locatable(ax3)
# # vmax=rec.max()
# # imax1 = ax1.imshow(im1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
# # imax2 = ax2.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
# # imax3 = ax3.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
# #
# # ax1.set_title(title[0])
# # ax2.set_title(title[1])
# # ax3.set_title(title[2])
# #
# # for ax in [ax1, ax2, ax3]:
# #     ax.get_xaxis().set_ticks([])
# #     ax.get_yaxis().set_ticks([])
# #     ax.grid(False)
# #
# # if scale is not None:
# #     fontprops = fm.FontProperties(size=12)
# #     scalebar = AnchoredSizeBar(ax1.transData,
# #                                scale[0], scale[1], 'lower right',
# #                                pad=0.1,
# #                                color='white',
# #                                frameon=False,
# #                                size_vertical=im1.shape[0] / 40,
# #                                fontproperties=fontprops)
# #
# #     ax1.add_artist(scalebar)
# # plt.tight_layout(pad=0)
# # plt.show()
# # #%%
# # def animate(i):
# #     imax1.set_data(rec[i, :, :])
# #     imax2.set_data(rec[:, i, :])
# #     imax3.set_data(rec[:, :, i])
# #
# # lin_ani = animation.FuncAnimation(fig, animate, frames=np.arange(rec.shape[0]), repeat=True)
# #
# # FFwriter = animation.FFMpegWriter(fps=10)
# # lin_ani.save(fig_path + 'merge2.mp4', writer=FFwriter)
# %%
path1 = '/home/philipp/drop/Public/'
np.save(path1 + 'mengyu_best.npy', sol.cpu().numpy())
