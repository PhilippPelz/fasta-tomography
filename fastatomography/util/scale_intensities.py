import numpy as np
import torch as th
from h5py import h5r
from numpy.linalg import norm
from tqdm import trange
from util import plot
from h5rw import h5read, h5write
import matplotlib.pyplot as plt

path = '/home/philipp/projects2/tomo/2019-09-09_kate_pd/'
fn = 'kate_bg_pad_align.npy'
d = np.load(path + fn)
d = np.transpose(d, (2, 1, 0))
d = d[:,400:1200,400:1200]
# %%

s = np.sum(d, 1)

f, a = plt.subplots(figsize=(15, 12))
# s = s[[25, 0]]
for si in s:
    a.plot(np.arange(len(si)), si)
plt.show()

# %%

d1 = th.from_numpy(d).float()
# %%
from torch.nn.functional import mse_loss
from torch.optim import Adam

refind = 25
ref = th.sum(d1[refind], 1).float()
f, a = plt.subplots(figsize=(15, 12))
a.plot(np.arange(len(si)), ref.numpy())
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
s = th.sum(d2 * fac.unsqueeze(1).unsqueeze(1).expand_as(d2), 1).float()
# s *= fac.unsqueeze(1).expand_as(s)
for si in s:
    a.plot(np.arange(len(si)), si.detach().numpy())
a.plot(np.arange(len(si)), ref.numpy(), linewidth=5)
plt.show()
# %%
f, a = plt.subplots(figsize=(15, 12))
a.scatter(np.arange(len(fac)), fac.detach().numpy(), linewidth=5)
plt.show()
# %%
for i in np.arange(24, 27):
    plot(d[i], f"{i}")
# %%
from numpy.fft import fftfreq, fft2, fftshift, ifft2

d2_corrected = d2 * fac.unsqueeze(1).unsqueeze(1).expand_as(d2)
d2_corrected = d2_corrected.detach().numpy()
for i, d in enumerate(d2_corrected):
    plot(d, f"{i}")

# %%

h5write(path + '2020-01-09_kate_intensities_corrected.h5', data=d2_corrected)

# %%
d2s = np.sum(d2_corrected, 0)
plot(d2s)

from scipy.ndimage.measurements import center_of_mass

x = (d2s > 1e5).astype(np.float32)
com = center_of_mass(x)
print(com)

c = np.array(com).astype(np.int)

s = 350
f, a = plt.subplots()
a.imshow(d2s[c[0] - s:c[0] + s, c[1] - s:c[1] + s])
# a.scatter(com[1], com[0])
plt.show()

d2c = d2_corrected[:, c[0] - s:c[0] + s, c[1] - s:c[1] + s]
#%%
i = 2
x = np.log10(np.abs(fftshift(fft2(d2c[i]))))
plot(d2c[i], f"{i}")
plot(x, f"{i}")
# %%

h5write(path + '2019-10-07_cropped.h5', data=d2c)


# %%
def rebin(a, s, mode='sum'):
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,(3,2))
    >>> a=rand(6); b=rebin(a,2)
    '''
    shape = np.asarray(a.shape)
    lenShape = len(shape)
    args = np.ones_like(shape) * np.asarray(s)
    factor = shape // args
    s1 = tuple()
    for i in range(lenShape):
        s1 += (factor[i],)
        s1 += (args[i],)
    ret = a.reshape(s1)
    for i in range(lenShape):
        j = (lenShape - i) * 2 - 1
        ret = ret.sum(j)
    if mode == 'mean':
        ret /= np.prod(args)
    return ret


d2c_bin2 = rebin(d2c, (1, 2, 2))
h5write(path + '2019-10-07_cropped_bin2.h5', data=d2c)
#%%
from scipy.io import savemat
savemat(path + '2019-10-07_cropped.mat',{'data':d2c})
savemat(path + '2019-10-07_cropped_bin2.mat',{'data':d2c_bin2})
# %%


angles = np.array([6.403600311279296875 * 10,
                   6.220299911499023438 * 10,
                   6.005899810791015625 * 10,
                   5.808100128173828125 * 10,
                   5.605500030517578125 * 10,
                   5.404800033569335938 * 10,
                   5.205199813842773438 * 10,
                   5.005799865722656250 * 10,
                   4.805099868774414062 * 10,
                   4.607400131225585938 * 10,
                   4.413700103759765625 * 10,
                   4.203799819946289062 * 10,
                   3.703699874877929688 * 10,
                   3.407500076293945312 * 10,
                   3.101199913024902344 * 10,
                   2.804999923706054688 * 10,
                   2.505400085449218750 * 10,
                   2.206399917602539062 * 10,
                   1.902499961853027344 * 10,
                   1.605599975585937500 * 10,
                   1.306299972534179688 * 10,
                   1.006599998474121094 * 10,
                   7.063000202178955078,
                   4.046999931335449219,
                   1.054999947547912598,
                   -1.904000043869018555,
                   -4.939000129699707031,
                   -7.941999912261962891,
                   -1.093700027465820312 * 10,
                   -1.393400001525878906 * 10,
                   -1.691099929809570312 * 10,
                   -1.995299911499023438 * 10,
                   -2.291099929809570312 * 10,
                   -2.691399955749511719 * 10,
                   -2.993199920654296875 * 10,
                   -3.291099929809570312 * 10,
                   -3.590999984741210938 * 10,
                   -3.890200042724609375 * 10,
                   -4.190999984741210938 * 10,
                   -4.389799880981445312 * 10,
                   -4.594599914550781250 * 10,
                   -4.793799972534179688 * 10,
                   -4.991500091552734375 * 10,
                   -5.191799926757812500 * 10,
                   -5.391600036621093750 * 10,
                   -5.591099929809570312 * 10,
                   -5.791299819946289062 * 10,
                   -5.988399887084960938 * 10,
                   -6.190499877929687500 * 10
                   ])

savemat(path + 'angles.mat', {'angles':angles})

# %%
fd = fftshift(np.log10(np.abs(fft2(d2_corrected))), (1, 2))
# %%
c = fd.shape[1] // 2
s = 150
fd1 = fd[:, c - s:c + s, c - s:c + s]
plot(fd1[25])
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

savePath = None
cmap = ['inferno', 'inferno']
title = ['Tilt %02d' % 0, 'FFT if tilt %02d' % 0]
show = True
figsize = (10, 10)
pix = 100
q = 0.1736 * pix
scale = (pix, f'{q / 10} nm')

im1, im2 = d2c[0], fd1[0]
fig = plt.figure(figsize=figsize)
gs1 = gridspec.GridSpec(1, 2)
gs1.update(wspace=0, hspace=0.2)  # set the spacing between axes.
ax0 = plt.subplot(gs1[0, 0])
ax1 = plt.subplot(gs1[0, 1])

imax0 = ax0.imshow(im1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))
imax1 = ax1.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))

ax0.set_title(title[0])
ax1.set_title(title[1])

for ax in [ax0, ax1]:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticklabels([])

if scale is not None:
    fontprops = fm.FontProperties(size=18)
    scalebar = AnchoredSizeBar(ax0.transData,
                               scale[0], scale[1], 'lower right',
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=im1.shape[0] / 40,
                               fontproperties=fontprops)

    ax0.add_artist(scalebar)

# patch = patches.Circle((14.5, 14.5), radius=14, transform=ax0.transData, fill=None, color='r')
# ax0.add_patch(patch)

ax0.grid(False)
ax1.grid(False)

plt.tight_layout()

if show:
    plt.show()


# %%
def animate(ind):
    imax0.set_data(d2c[ind])
    imax1.set_data(fd1[ind])
    title = ['Tilt %02d' % ind, 'FFT if tilt %02d' % ind]
    ax0.set_title(title[0])
    ax1.set_title(title[1])


lin_ani = animation.FuncAnimation(fig, animate, frames=np.arange(d2_corrected.shape[0]), repeat=True)

FFwriter = animation.FFMpegWriter(fps=25)
lin_ani.save('/home/philipp/projects2/tomo/2019-09-09_kate_pd/02_denoise/cropped.mp4', writer=FFwriter)
