from fastatomography.util import *
#%%
path = '/home/philipp/projects2/tomo/2019-03-18_Pd_loop/'
#%%
# path = '/home/philipp/projects2/tomo/2019-04-17-Pd_helix/philipp/'
# fn = 'RecFISTA_reg5.npy'

# rec = np.load(path + fn)
mask = np.load(path + 'mask2.npy')
mask = np.transpose(mask, (1, 0, 2))
#
#
# # %%
# blur1 = blur.copy()
# # blur1[:,:60,:] = 0
# # blur1[:,380:,:] = 0
#
# # plot(blur1[:, :, 100])
# # plot(blur1[:, :, 200])
# # plot(blur1[:, :, 300])
# # plot(blur1[:, :, 400])
#
# for i in range(20):
#     plot(blur1[i * 20, :, :])
# # plot(blur1[:, 200, :])
#
#
# # %%
# # plot(blur1[:, 200, :])
# from skimage import io
#
# im = io.imread('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/rec0.tiff')
# print(im.shape)

# im = np.transpose(im, (2, 1, 0))
# io.imsave('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/rec0T.tiff', im)
# %%
from scipy.ndimage import zoom

s = np.array(mask.shape)
m2 = np.zeros(2 * s)
m2 = zoom(mask, 2)
# for i, slice in enumerate(mask):
#     print(f"{i}/{mask.shape[0]}")
#     m2[i] = zoom(mask[i],2)
# from skimage import io

# im = io.imread('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/mask.tiff')
# print(im.shape)
# %%
# im = np.transpose(im, (2, 1, 0))
# print(im.shape)
# io.imsave('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/mask.tiff', im)

# %%
# mask = (im < 1.1e-16).astype(np.float)
# mask = np.transpose(mask, [2, 1, 0])
# %%
# mask = np.zeros_like(rec)
# %%
# mask3[0] = mask3[70]
ms = np.sum(mask3, (1, 2))

drawn = ms > 10

# drawn2 = np.logical_and(np.arange(len(ms))>100,ms > 20000)

# drawn3 = np.logical_or(drawn,drawn2)

f, a = plt.subplots()
a.plot(np.arange((len(ms))), ms)
# a.plot(np.arange((len(ms))),drawn3.astype(np.float)*4e4)
a.plot(np.arange((len(ms))), drawn.astype(np.float) * 3.8e4)
# a.plot(np.arange((len(ms))),drawn2.astype(np.float)*3e4)
plt.show()

# %%
from tqdm import trange

mask2 = mask3.copy()
for i in trange(mask3.shape[0]):
    if not drawn[i]:
        for j in range(i):
            if drawn[i - (j+1)]:
                mask2[i] = mask3[i - (j+1)]
                break
# %%

plot(mask2[:, 200, :])

# %%
# for i in trange(100):
#     plot(mask2[i])
# %%
# mask2 = np.transpose(mask2, [2, 1, 0])
# %%
# io.imsave('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/rec0TmaskT2.tiff', mask2)
# %%
# np.save('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/rec0TmaskT2.npy', mask2)

# %%

mask2[199:] = 0
np.save(path + 'mask3.npy', mask2.astype(np.float))

# %%
mask = np.zeros_like(im)
io.imsave('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/mask2.tiff', mask2.astype(np.int))

# %%
from scipy.io import loadmat
mask3 = loadmat('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/mask.mat')['d']
mask3 = np.transpose(mask3,(1,0,2))
#%%
mask3 = np.load('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/mask.npy')
# %%
mask3 = mask2.astype(np.float)
# mask3[mask3 < 0.05] = 0.4
plot(mask3[:, 100, :])
from scipy.ndimage import gaussian_filter
from scipy.io import savemat

# mask2[431:] = 0
mask3 = gaussian_filter(mask3, 7)
plot(mask3[:, 100, :])
# mask3 += 0.7
mask3 /= mask3.max()
plot(mask3[:, 100, :])
mask3= (mask3>0.4).astype(np.float32)
plot(mask3[:, 100, :])

#%%
# mask4 = np.transpose(mask3, (1, 0, 2))
mask4 = np.transpose(mask3,(1,0,2))
# mask4 = mask3
savemat(path+'thresh_mask', {'m': mask4.astype(np.float32)})
# %%
np.save(path+'mask_0p7.npy', mask3)
