import matplotlib as mpl
from numpy.random import uniform
from scipy.ndimage import fourier_shift
from skimage.feature import register_translation
from tqdm import trange

# mpl.rcParams['text.usetex'] = True
from fastatomography.fasta_tomography import fasta_tomography_nonnegative_shrink
from fastatomography.default_dependencies import *
from scipy.io import loadmat
from skimage.filters import gaussian, median
from fastatomography.util import *
from fastatomography.tomo import ray_transforms
from fastatomography.util.plotting import plot_rotating_point, plot_translations, save_stack_movie, save_stack_gif
from tifffile import imsave, imwrite, imread

# load data
path = '/home/philipp/nvme/2020-09-22/smatrix/'
fig_path = '/home/philipp/drop/Public/2020-03-30_team/'
fn = 'fully_aligned.tif'
angles_fn = 'angles1.mat'
dx = 0.33405737560941523

proj = imread(path+fn)
n_angles = 3
angles = np.zeros((3, n_angles), dtype=np.float32)
angles[0, :] = 0
angles[1, :] = np.deg2rad([0,5,10])
angles[2, :] = 0

# %%
pstack = proj  # np.transpose(proj, (2, 1, 0))
N = np.array(pstack.shape[1:])
n_angles = pstack.shape[0]
plotmosaic(pstack, 'Tilt series overview', dpi=600, savePath=f'{fig_path}series_overview')
# pad in real space to alleviate aliasing artifacts
pstack = np.pad(pstack, ((0, 0), (N[0] // 3, N[0] // 3), (N[0] // 3, N[0] // 3)), mode='constant', constant_values=0)
N = np.array(pstack.shape[1:])
# %%
save_stack_gif(path + 'gif', pstack, np.rad2deg(angles[1, :]), dx)
#%%
from scipy.signal.windows import tukey
from skimage.transform import rotate
rot2 = -50.066166
srot2 = np.array([rotate(si,rot2, resize=True) for si in pstack])
m = 620
crop = srot2[2,m-10:-m-30,m+100:-m-20]
plot(crop, dpi=(300))
#%%
m = np.mean(crop,1)
# fig, ax = plt.subplots()
# ax.scatter(np.arange(len(m)),m)
# plt.show()
change = np.abs(m)>0.02
# crop[change,:] -= m[change,None]
crop[:] -= m[:,None]
crop += 5
srot2[2, srot2[2] < 4.6] = 0
crop -= 5
#%%
cs = crop.shape
w1 = np.zeros_like(crop)
w2 = np.zeros_like(crop)
w1[:] = tukey(cs[0])[:, None]
w2[:] = tukey(cs[1])[None, :]
w = w1 * w2
crop *= w
#%%p
plot(srot2[2], dpi=(300))
#%%
m = 560
crop = srot2[1,m-10:-m-30,m+100:-m-20]
cs = crop.shape
w1 = np.zeros_like(crop)
w2 = np.zeros_like(crop)
w1[:] = tukey(cs[0])[:, None]
w2[:] = tukey(cs[1])[None, :]
w = w1 * w2
crop *= w
plot(crop, dpi=(300))
#%%
plot(srot2[1], dpi=(300))
#%%
m = 520
crop = srot2[0,m+160:-m-30,m+180:-m-80]


m = np.mean(crop,1)
fig, ax = plt.subplots()
ax.scatter(np.arange(len(m)),m)
plt.show()

change = np.abs(m)>0.01
crop[change,:] -= m[change,None]

m = 520
crop = srot2[0,m+160:-m-30,m+160:-m-80]

cs = crop.shape
w1 = np.zeros_like(crop)
w2 = np.zeros_like(crop)
w1[:] = tukey(cs[0])[:, None]
w2[:] = tukey(cs[1])[None, :]
w = w1 * w2
crop *= w

plot(crop, dpi=(300))
#%%
plot(srot2[0], dpi=(300))
#%%
fig, ax = plt.subplots()
ax.hist(crop.ravel(),bins=100)
plt.show()
#%%
# from skimage.morphology import disk
# gpstack = np.array([ps - gaussian(ps,3) for ps in pstack])
# gpstack = np.array([ps - median(ps,disk(2)) for ps in pstack])
# plot(gpstack[2], dpi=(300))
rot2 = 50.066166
gpstack = np.array([rotate(si,rot2, resize=True) for si in srot2])
plotmosaic(gpstack, 'gpstack series overview', dpi=600, savePath=f'{fig_path}gpstack')
# %% do angle refinement on low-resolution data
opts = Param()
opts.record_objective = True
opts.verbose = True
opts.string_header = '     '
opts.max_iters = 50
opts.tol = 1e-5
opts.accelerate = True
opts.adaptive = False
opts.restart = False
# step size, is picked automatgically if you leave this out. Sometimes the convergence curve looks weird,
# and you have to set it manually
# opts.tau = 20

#
refinement_iterations = 10
i_refine = 0
refinement_objectives = np.zeros((refinement_iterations,))
angle_trial_start_range = np.deg2rad(0.3)
angle_trials = 6

i_res = 1
stack_res = gpstack.copy()
m = 980
stack_res = stack_res[:, m:-m, m:-m]

stack_res -= stack_res.min()

N_res = gpstack.shape[1:]
registration_upsample_factor = 10
resolution_upsample_factor = 1
translation_shifts = np.zeros((n_angles, 2))

# import h5py
# with h5py.File('/home/philipp/projects2/tomo/PdDecahedralTomo/05_unbiased_pruning/stacks.h5','a') as f:
#     f.create_dataset(f'stack_res1',data=stack_res)
#
plot(np.sum(stack_res, 0), 'sum(stack res)', dpi=400)
 #%%
ss = np.sum(stack_res, 2)
fig, ax = plt.subplots()
for i, si in enumerate(ss):
    ax.plot(si, label =f'{i}')
plt.legend()
plt.show()
 #%%

plotmosaic(stack_res, 'Tilt series overview', dpi=600, savePath=f'{fig_path}series_overview')

# %%
ps = stack_res.shape
projection_shape = ps[1:]
vol_shape = (projection_shape[0], projection_shape[1], projection_shape[1])
real_space_extent = np.array([projection_shape[0], projection_shape[0], projection_shape[1]])

A, At = ray_transforms(real_space_extent, projection_shape, n_angles, interp='nearest')
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
opts.tau = 1
best_angles = angles
mu = 1e-6

x0 = th.zeros(*vol_shape, dtype=th.float32).cuda()
y = th.as_tensor(np.transpose(stack_res, (1, 0, 2))).contiguous().float().cuda()
print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing ...')
sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts)
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
print(f'best R-factor: {out.R_factors[-1]}')
#%%
mean = rec.mean()
max = rec.max()
dd = max - mean

rec1 = np.clip(rec,mean-dd,mean+dd)
#%%
from tifffile import imsave, imwrite, imread
m = 50
imwrite(path + 'recon2.tif', np.transpose(rec1[m:-m,:,m:-m],(1,2,0)).astype('float32'), imagej=True, resolution=(1./(1/10), 1./(1/10)),
            metadata={'spacing': 1 / 10, 'unit': 'nm', 'axes': 'ZYX'})
# %% refinement iteration
sh = stack_res.shape
pad_width = ((sh[1] // 4, sh[1] // 4), (sh[2] // 4, sh[2] // 4))
for i_refine in range(refinement_iterations):
    x0 = th.zeros(*vol_shape, dtype=th.float32).cuda()
    y = th.as_tensor(np.transpose(stack_res, (1, 0, 2))).contiguous().float().cuda()
    print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing ...')
    # sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts)
    # refinement_objectives[i_refine] = out.objective[-1]
    # # register projection with subpixel precision
    # print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: subpixel registration ...')
    # y_model = th.zeros_like(y)
    # y_model = A(sol, out=y_model, angles=best_angles)
    # y_model = th.transpose(y_model, 0, 1).cpu().numpy()
    precision = registration_upsample_factor * resolution_upsample_factor
    # rms_translation_errors = np.zeros((n_angles,), dtype=np.float32)
    # current_shifts = np.zeros((n_angles, 2))
    # for i in range(n_angles):
    #     shift, rms_translation_errors[i], diffphase = register_translation(y_model[i], stack_res[i], precision)
    #     # print(f'angle {i:-4d} shift: {shift}')
    #     translation_shifts[i] += shift
    #     current_shifts[i] = shift
    #     stack_res[i] = ifftn(fourier_shift(fftn(stack_res[i]), shift))
    # print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: max. shift: {np.max(np.abs(current_shifts))}')
    # # plot_translations(current_shifts, f'Translations iteration {i_refine}',
    # #                   savePath=f'{fig_path}{i_refine:03d}_translations')
    # # plotmosaic(np.abs(y_model - stack_res), f'Tilt series residual after translations iteration {i_refine}',
    # #            dpi=600, savePath=f'{fig_path}{i_refine:03d}_tilt_series_residuals')
    #
    # print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing for angle refinement ...')
    sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts)
    print(f'best R-factor: {out.R_factors[-1]}')
    # %
    # compute trial angle losses and find best
    angle_trial_range = (1 - (i_refine / refinement_iterations)) * angle_trial_start_range
    print(
        f'it {i_refine:-4d}/{refinement_iterations:-4d}: angle refinement with range {np.rad2deg(angle_trial_range):2.2f} deg ...')
    trial_losses = np.zeros((angle_trials + 1, n_angles))
    trial_angles = np.zeros((angle_trials + 1, *best_angles.shape))
    trial_shifts = np.zeros((angle_trials + 1, n_angles, 2))
    for t in range(angle_trials):
        random_offsets = uniform(-angle_trial_range, angle_trial_range, best_angles.shape)
        # random_offsets[0] = 0
        # random_offsets[2] = 0
        trial_angles[t] = random_offsets + best_angles
        trial_proj = th.zeros_like(y)
        trial_proj = A(sol, out=trial_proj, angles=trial_angles[t])

        for n in range(n_angles):
            yn = np.pad(y[:, n, :].cpu().numpy(), pad_width)
            y_modeln = np.pad(trial_proj[:, n, :].cpu().numpy(), pad_width)
            shift, _, diffphase = register_translation(yn, y_modeln, precision)
            trial_shifts[t, n] = shift
            tt = ifftn(fourier_shift(fftn(y_modeln), shift)).real
            trial_proj[:, n, :] = th.as_tensor(
                tt[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]).cuda()

        trial_losses[t] = (th.norm((trial_proj - y).cpu(), dim=(0, 2)) ** 2).numpy()
        trial_angles[-1] = best_angles
    trial_proj = th.zeros_like(y)
    trial_proj = A(sol, out=trial_proj, angles=trial_angles[t])
    trial_losses[-1] = (th.norm((trial_proj - y).cpu(), dim=(0, 2)) ** 2).numpy()
    min_loss = np.min(trial_losses, axis=0)
    mini_ind = np.argmin(trial_losses, axis=0)
    print(f'min indices: {mini_ind}')

    # pick angles with minimum error
    best_angles = np.squeeze(
        np.stack([a[mini_ind[i]] for i, a in enumerate(np.split(trial_angles, np.arange(1, n_angles), axis=2))])).T
    shifts = np.squeeze(
        np.stack([a[mini_ind[i]] for i, a in enumerate(np.split(trial_shifts, np.arange(1, n_angles), axis=1))]))
    for i in range(n_angles):
        s = np.pad(stack_res[i], pad_width)
        s = ifftn(fourier_shift(fftn(s), -shifts[i]))
        stack_res[i] = s[pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]
        translation_shifts[i] += -shifts[i]
    # print(f'best angles: {np.rad2deg(best_angles[1])}')

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
print(f'best R-factor: {out.R_factors[-1]}')
#%%
import h5py
with h5py.File('/home/philipp/projects2/tomo/PdDecahedralTomo/05_unbiased_pruning/angles.h5','a') as f:

    f.create_dataset('angles',data=best_angles)

# %%
plot_rotating_point(best_angles, f'Best angles iteration {i_refine}', savePath=f'{fig_path}best_angles.png', dpi=300)
plot_rotating_point(angles, f'Start angles iteration {i_refine}', savePath=f'{fig_path}start_angles.png', dpi=300)
# %%
# stack_res2 = res_stacks[-1].copy()
# for i in range(n_angles):
#     stack_res2[i] = ifftn(fourier_shift(fftn(stack_res2[i]), translation_shifts[i] * upsample_factors[i_res]))
# # %%
# plotmosaic(stack_res2, dpi=600, savePath=f'{fig_path}stack_aligned_fullres.png')
# # %%
# m = 160
# stack_res3 = stack_res2[:, m:-m, m:-m]
# plot(np.sum(stack_res3, 0), 'sum(stack res)')
# # %%
# # stack_res3 = stack_res
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
# opts.max_iters = 200
# opts.tol = 1e-5
# opts.accelerate = True
# opts.adaptive = True
# opts.restart = True
# opts.tau = 20
# mu = 100e-2
#
# A, At = ray_transforms(real_space_extent, projection_shape, n_angles, interp='linear')
# x0 = th.zeros(*vol_shape, dtype=th.float32).cuda()
# y = th.as_tensor(np.transpose(stack_res3, (1, 0, 2))).contiguous().float().cuda()
# print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing ...')
# sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts)
# # %%
# mp = th.zeros_like(y)
# mp = A(sol, out=mp, angles=best_angles)
# i = 32
#
# plot(np.hstack([y[:, i, :].cpu(), mp[:, i, :].cpu()]), 'y                   y_model')
# # %%
# diff = np.transpose((y - mp).cpu().numpy(), (1, 0, 2))
# plotmosaic(diff, 'y - y_model', dpi=900)
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
# %%
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
# path1 = '/home/philipp/drop/Public/'
# title = ['x-z', 'y-z', 'y-x']
# cmap = 'viridis'
# scale = (40 / dx, '4 nm')
#
# im1, im2, im3 = [rec[0, :, :], rec[:, 0, :], rec[:, :, 0]]
# fig = plt.figure(dpi=600)
# gs1 = gridspec.GridSpec(1, 3)
# gs1.update(wspace=0, hspace=0)  # set the spacing between axes.
# ax1 = plt.subplot(gs1[0, 0])
# ax2 = plt.subplot(gs1[0, 1])
# ax3 = plt.subplot(gs1[0, 2])
# div1 = make_axes_locatable(ax1)
# div2 = make_axes_locatable(ax2)
# vmax=rec.max()
# imax1 = ax1.imshow(im1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
# imax2 = ax2.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
# imax3 = ax3.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
#
# ax1.set_title(title[0])
# ax2.set_title(title[1])
# ax3.set_title(title[2])
#
# for ax in [ax1, ax2, ax3]:
#     ax.get_xaxis().set_ticks([])
#     ax.get_yaxis().set_ticks([])
#     ax.grid(False)
#
# if scale is not None:
#     fontprops = fm.FontProperties(size=12)
#     scalebar = AnchoredSizeBar(ax1.transData,
#                                scale[0], scale[1], 'lower right',
#                                pad=0.1,
#                                color='white',
#                                frameon=False,
#                                size_vertical=im1.shape[0] / 40,
#                                fontproperties=fontprops)
#
#     ax1.add_artist(scalebar)
# plt.tight_layout(pad=0)
# plt.show()
# def animate(i):
#     imax1.set_data(rec[i, :, :])
#     imax2.set_data(rec[:, i, :])
#     imax3.set_data(rec[:, :, i])
#
# lin_ani = animation.FuncAnimation(fig, animate, frames=np.arange(rec.shape[0]), repeat=True)
#
# FFwriter = animation.FFMpegWriter(fps=10)
# lin_ani.save(fig_path + 'merge2.mp4', writer=FFwriter)
#%%
mus = [0.125,0.25,0.5,1,2,4]
sols = []
Rs = []
for mu in mus:
    best_angles = angles
    opts = Param()
    opts.record_objective = True
    opts.verbose = True
    opts.string_header = '     '
    opts.max_iters = 50
    opts.tol = 1e-5
    opts.accelerate = True
    opts.adaptive = True
    opts.restart = True
    # step size, is picked automagically if you leave this out. Sometimes the convergence curve looks weird,
    # and you have to set it manually
    opts.tau = 25
    x0 = th.zeros(*vol_shape, dtype=th.float32).cuda()
    y = th.as_tensor(np.transpose(stack_res, (1, 0, 2))).contiguous().float().cuda()
    print(f'it {i_refine:-4d}/{refinement_iterations:-4d}: reconstructing ...')
    sol, out, opts_out = fasta_tomography_nonnegative_shrink(A, At, x0, y, mu, best_angles, opts)
    sols.append(sol.cpu().numpy())
    Rs.append(out.R_factors[-1])
Rs = np.array(Rs)
# %%
save_path = path
for mu1, sol1, R1 in zip(mus,sols,Rs):
    np.save(save_path + fn.split('.')[0] + '_recon_' + f'mu_{mu1*100:2.1f}_'+ f'R_{R1*100:2.1f}_'+'.npy', sol1)
