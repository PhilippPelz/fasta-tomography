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
from skimage.transform import rotate

# load data
path = '/home/philipp/projects2/tomo/2019-09-09_kate_pd/'
fig_path = '/home/philipp/drop/Public/2020-03-30_team/'
fn = 'Rotation_34.emd'

f = h5read(path + fn, 'data/drift/data')['data/drift/data']

p0 = np.abs(fftshift(fft2(f[0])))
p1 = np.abs(fftshift(fft2(f[1])))

sigma = 2

gf0 = gaussian(f[0], sigma)
gf1 = gaussian(f[1], sigma)

rgf1 = rotate(gf1, -90)

plot(gf0, '0')
plot(rgf1, '90')

# plot(np.log10(p0), 'p0')
# plot(np.log10(p1), 'p1')

#%%
import math
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import transform as tf

tform = tf.SimilarityTransform(scale=1, rotation=math.pi/2,
                               translation=(0, 1))
print(tform.params)
coord = [1, 0]

matrix = tform.params.copy()
matrix[1, 2] = 2
tform2 = tf.SimilarityTransform(matrix)

print(tform2(coord))
print(tform2.inverse(tform(coord)))

text = data.text()

tform = tf.SimilarityTransform(scale=1, rotation=math.pi/4,
                               translation=(text.shape[0]/2, -100))

rotated = tf.warp(text, tform)
back_rotated = tf.warp(rotated, tform.inverse)

fig, ax = plt.subplots(nrows=3)

ax[0].imshow(text, cmap=plt.cm.gray)
ax[1].imshow(rotated, cmap=plt.cm.gray)
ax[2].imshow(back_rotated, cmap=plt.cm.gray)

for a in ax:
    a.axis('off')

plt.tight_layout()

#%%
text = data.text()

src = np.array([[0, 0], [0, 50], [300, 50], [300, 0]])
dst = np.array([[155, 15], [65, 40], [260, 130], [360, 95]])

tform3 = tf.ProjectiveTransform()
tform3.estimate(src, dst)
warped = tf.warp(text, tform3, output_shape=(50, 300))

fig, ax = plt.subplots(nrows=2, figsize=(8, 3))

ax[0].imshow(text, cmap=plt.cm.gray)
ax[0].plot(dst[:, 0], dst[:, 1], '.r')
ax[1].imshow(warped, cmap=plt.cm.gray)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()