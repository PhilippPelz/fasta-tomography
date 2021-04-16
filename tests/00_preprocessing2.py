from scipy.ndimage import fourier_shift
from skimage.feature import register_translation
from skimage.filters import gaussian
from skimage.transform import rotate
from torch.nn.functional import mse_loss
from torch.optim import Adam
from tqdm import trange
from scipy.ndimage.measurements import center_of_mass
from fastatomography.util import *
from fastatomography.util.plotting import save_stack_gif

# A angles
# R scan rotations
# NY, NX image size

path = '/home/philipp/projects2/tomo/2019-09-09_kate_pd/'
fn = '20190909_BoxCd4_Pd_tomo3.emd'
name = fn.split('.')[0]
log_filename = f'{name}.log'

setup_logging(path, log_filename)

logging.info(f'Loading file {path + fn}')
d = h5read(path + fn)
# %%
# A x R x NY x NX
data = d['data']['raw']['data']
# 6 x A * R
image_params = d['data']['raw']['imageParameters']
image_set = image_params[0]
rep_number = image_params[1]
scan_rotation = image_params[2]
dx = np.array([image_params[3], image_params[4]])[0, 0]
time = image_params[5]
# 6 x A
image_set_params = d['data']['raw']['imageSetParameters']
stage_pos = image_set_params[0:3]
alpha = image_set_params[3]
gamma = image_set_params[4]

logging.info(f'Initial data shape:  {data.shape}')
logging.info(f'pixel size:          {dx * 1e12}pm')
logging.info(f'scan rotations:      {np.rad2deg(scan_rotation[0])}')

# %%
stack_aligned = []

for i in trange(data.shape[0]):
    d2 = rotate(data[i, 1], 90, resize=False, center=None, order=1, mode='constant', cval=0, clip=True,
                preserve_range=False)

    shift, error, diffphase = register_translation(data[i, 0], d2, upsample_factor=20, space='real')
    d2_aligned = ifftn(fourier_shift(fftn(d2), shift)).real

    aligned = data[i, 0] + d2_aligned

    stack_aligned.append(aligned)

stack_aligned = np.array(stack_aligned)
# %%
plotmosaic(stack_aligned, 'Stack after alignment', dpi=600)
#%%
bad = []
stack_aligned = np.delete(stack_aligned, bad, axis=0)
alpha2 = np.delete(alpha, bad, axis=0)
gamma2 = np.delete(gamma, bad, axis=0)
stage_pos2 = np.delete(stage_pos, bad, axis=1)
# %%
sh = stack_aligned.shape
c = sh[0] // 2
plot(stack_aligned[c])

com = center_of_mass(gaussian(stack_aligned[c], 10))
print(com)
centered = ifftn(fourier_shift(fftn(stack_aligned[c]), (-20, -50))).real

f, a = plt.subplots()
a.imshow(centered)
plt.vlines(sh[2] // 2, 0, sh[1])
plt.hlines(sh[1] // 2, 0, sh[2])
plt.show()
# %%
stack_aligned[c] = centered
com = center_of_mass(gaussian(stack_aligned[c], 10))
print(com)
# %% rough cross correlation alignment
for i in trange(c, 0, -1):
    shift, error, diffphase = register_translation(stack_aligned[i], stack_aligned[i - 1], upsample_factor=2,
                                                   space='real')
    stack_aligned[i - 1] = ifftn(fourier_shift(fftn(stack_aligned[i - 1]), shift)).real

for i in trange(c, stack_aligned.shape[0] - 1):
    shift, error, diffphase = register_translation(stack_aligned[i], stack_aligned[i + 1], upsample_factor=2,
                                                   space='real')
    stack_aligned[i + 1] = ifftn(fourier_shift(fftn(stack_aligned[i + 1]), shift)).real
plotmosaic(stack_aligned, 'Stack after alignment', dpi=600)
# %%
crop_margins = m = 10
plot(stack_aligned[0])

stack_cropped = stack_aligned[:, m:-m, m:-m]
plotmosaic(stack_cropped, dpi=600)
# %%
d1 = th.from_numpy(stack_cropped).float()
refind = sh[0]//2
ref = th.sum(d1[refind], 1).float()
f, a = plt.subplots(figsize=(15, 12))
s = th.sum(d1, 1).float()
# s *= fac.unsqueeze(1).expand_as(s)
for si in s:
    a.plot(np.arange(len(si)), si.detach().numpy())
a.plot(np.arange(ref.shape[0]), ref.numpy(), linewidth=5)
plt.show()
# %%
d2 = d1
fac = th.ones((d1.shape[0]), requires_grad=True)
last_loss = 0
opt = Adam([fac], lr=1e-2)

ds = th.sum(d2, 1).float()
for i in range(100):
    opt.zero_grad()

    s = ds * fac.unsqueeze(1).expand_as(ds)

    loss = mse_loss(s, ref)

    # f, a = plt.subplots(figsize=(15, 12))
    # for si in s:
    #     a.plot(np.arange(len(si)), si.detach().numpy())
    # a.plot(np.arange(len(si)), ref.numpy(), linewidth=5)
    # plt.show()

    loss.backward()

    print(f"i: {i} L = {loss.item():3.6g} dL = {last_loss - loss.item():3.3g}")

    opt.step()
    # print(fac)

    last_loss = loss.item()
    # fac[j] = fac2.detach().item()
print(fac)
# %%
f, a = plt.subplots(figsize=(15, 12))
s = th.sum(d1, 1).float()
s *= fac.unsqueeze(1).expand_as(s)
for si in s:
    a.plot(np.arange(len(si)), si.detach().numpy())
a.plot(np.arange(ref.shape[0]), ref.numpy(), linewidth=5)
plt.show()
# %%
f, a = plt.subplots(figsize=(15, 12))
a.scatter(np.arange(len(fac)), fac.detach().numpy(), linewidth=5)
plt.show()

tol = 0.25
bad = (fac < 1 - tol).__or__(fac > 1 + tol)
print(f'bad indices, because intensity difference is too high: {np.where(bad.numpy())}')

d2_good = d2[~bad]
fac_good = fac[~bad]

alpha3 = alpha2[~bad]
gamma3 = gamma2[~bad]
stage_pos3 = stage_pos2[:, ~bad]

d2_corrected = d2_good * fac_good.unsqueeze(1).unsqueeze(1).expand_as(d2_good)
d2_corrected = d2_corrected.detach().numpy()

plotmosaic(d2_corrected, 'Intensity-corrected stack', dpi=600)
# %%
f, a = plt.subplots()
a.hist(d2_corrected.flatten(), bins=200)
plt.show()
# %%
stack_cropped_bg = d2_corrected - 33e3
stack_cropped_bg[stack_cropped_bg < 0] = 0

f, a = plt.subplots()
a.hist(stack_cropped_bg.flatten(), bins=200, range=(1, stack_cropped_bg.max()))
plt.show()

plotmosaic(stack_cropped_bg, 'Cropped and constant background subtracted', dpi=600)
# %% fine cross correlation alignment
sh = stack_cropped_bg.shape
pad_width = ((0, 0), (sh[1] // 2, sh[1] // 2), (sh[2] // 2, sh[2] // 2))
stack_cropped_bg = np.pad(stack_cropped_bg, pad_width)
c = stack_cropped_bg.shape[0] // 2
sigma = 10
for i in trange(c, 0, -1):
    shift, error, diffphase = register_translation(gaussian(stack_cropped_bg[i], sigma),
                                                   gaussian(stack_cropped_bg[i - 1], sigma), upsample_factor=5,
                                                   space='real')
    stack_cropped_bg[i - 1] = ifftn(fourier_shift(fftn(stack_cropped_bg[i - 1]), shift)).real

for i in trange(c, stack_cropped_bg.shape[0] - 1):
    shift, error, diffphase = register_translation(gaussian(stack_cropped_bg[i], sigma),
                                                   gaussian(stack_cropped_bg[i + 1], sigma), upsample_factor=5,
                                                   space='real')
    stack_cropped_bg[i + 1] = ifftn(fourier_shift(fftn(stack_cropped_bg[i + 1]), shift)).real

stack_cropped_bg = stack_cropped_bg[:, sh[1] // 2:-sh[1] // 2, sh[2] // 2:-sh[2] // 2]
stack_cropped_bg[stack_cropped_bg < 0] = 0
plotmosaic(stack_cropped_bg, 'Stack after alignment', dpi=600)
# %%

crop_margins = m = 200
plot(stack_cropped_bg[0])

stack_cropped_bg2 = stack_cropped_bg[:, m:-m, m:-m]

plotmosaic(stack_cropped_bg2, dpi=600)

#%%
duplicate_angle_threshold = 1e-1

ap1 = np.roll(alpha3, 1)
an1 = np.roll(alpha3, -1)

d1 = np.abs(ap1 - alpha3)
d2 = np.abs(an1 - alpha3)

d1s = d1 < duplicate_angle_threshold
d2s = d2 < duplicate_angle_threshold

duplicate_angles = np.where(d1s.__or__(d2s))[0]

for i in duplicate_angles:
    plot(stack_cropped_bg[i], f'stack_cropped_bg[{i}]')
#%%
bad =[10,29,51]
stack_aligned = np.delete(stack_cropped_bg, bad, axis=0)
alpha3 = np.delete(alpha3, bad, axis=0)
gamma3 = np.delete(gamma3, bad, axis=0)
stage_pos3 = np.delete(stage_pos3, bad, axis=1)
# %%
f, ax = plt.subplots(1, 2, dpi=150)
ax[0].set_title('Measured angles')
ax[0].scatter(alpha3, np.arange(len(alpha3)), s=1)
ax[0].set_ylabel('projection number')
ax[0].set_xlabel(r'$\alpha$ [degrees]')

ax[1].set_title('Measured angles')
ax[1].scatter(gamma3, np.arange(len(alpha3)), s=1)
ax[1].set_ylabel('projection number')
ax[1].set_xlabel(r'$\gamma$ [degrees]')

plt.show()
#%%
alpha3 = alpha3[np.argsort(alpha3)]
gamma3 = gamma3[np.argsort(alpha3)]
stack_aligned = stack_aligned[np.argsort(alpha3)]
stage_pos3 = stage_pos3[:,np.argsort(alpha3)]
#%%
save_stack_gif(path + 'cropped_aligned_fast', stack_aligned, alpha3, dx * 1e10, fps=10)
#%%
save_name = fn.split('.')[0]

save_dict = Param()

save_dict.stack = stack_aligned
save_dict.alpha = alpha3
save_dict.gamma = gamma3
save_dict.stage_pos = stage_pos3
save_dict.dx = dx

h5write(path+save_name+'_preprocessed.h5', save_dict)