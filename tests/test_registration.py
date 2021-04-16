from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
from skimage import data
from fastatomography.util import *

import numpy as np

n_shifts = 5

shifts = np.random.uniform(-5, 5, (5, 2))
shifts[-1] = -np.sum(shifts[:-1], 0)

x = data.camera().astype(np.float32)
x/=x.max()

x1 = x.copy()

plot(x)

print(np.sum(shifts, 0))
for i in range(n_shifts):
    x1 = fourier_shift(np.fft.fftn(x1), shifts[i])
    x1 = np.fft.ifftn(x1).real

plot(np.hstack([x,x1]))
#%%
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
from fastatomography.default_dependencies import *
from fastatomography.util import *
from skimage.filters import gaussian
from scipy import signal


image = data.camera()
N = np.array(image.shape)
M = N//2

# FWHM = 2*np.sqrt(2*np.log(2)) * sigma
# FWTM =  2*np.sqrt(2*np.log(10)) * sigma
res_factor = np.array([2,2])
sigma = (N/M)/(2*np.sqrt(2*np.log(2)))
kernel = np.outer(signal.gaussian(M[0], sigma[0]), signal.gaussian(M[1], sigma[1]))
fk = fft2(fftshift(kernel)).real
# plot(fftshift(kernel),'kernel')
# plot(kernel,'kernel')
# plot(fftshift(fk))

sigma = 5
x = np.linalg.norm(fourier_coordinates_2D(N,1/N, centered=True), axis=0)

gauss = np.exp(-x**2/(2*sigma**2))
gauss /= np.sum(gauss)

plot(fftshift(gauss),'gauss')
plot(fftshift(np.abs(ifft2(gauss, norm='ortho'))),'kernel')
# plot(ifft2(gauss).real)
#%%
sigma = 1
w = slepian_window(N, sigma)

fk = fft2(w, norm='ortho')
plot(w)
plot(fftshift(fk.real))
#%%
res_grid = np.prod(np.abs(fourier_coordinates_2D(N, 1 / N, centered=True)) < M[:, None, None] // 2,
                              axis=0).astype(np.bool)
falloff = 15
r1 = M[0]//4
resolution_masks = sector_mask(N, N // 2, r1 + falloff, (0, 360)).astype(np.float32)
resolution_masks = fftshift(gaussian(resolution_masks, falloff))
im_lowres = ifft2(((fft2(image)*resolution_masks)[res_grid]).reshape(M)).real
plot(resolution_masks)
plot(im_lowres,'lowres')

#%%

shift = (-22.4, 0)
# The shift corresponds to the pixel offset relative to the reference image
offset_image = fourier_shift(np.fft.fftn(image), shift)
offset_image = np.fft.ifftn(offset_image)
print(f"Known offset (y, x): {shift}")

# pixel precision first
shift, error, diffphase = register_translation(image, offset_image)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

# Show the output of a cross-correlation to show what the algorithm is
# doing behind the scenes
image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Cross-correlation")

plt.show()

print(f"Detected pixel offset (y, x): {shift}")

# subpixel precision
shift, error, diffphase = register_translation(image, offset_image, 100)

fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
ax3 = plt.subplot(1, 3, 3)

ax1.imshow(image, cmap='gray')
ax1.set_axis_off()
ax1.set_title('Reference image')

ax2.imshow(offset_image.real, cmap='gray')
ax2.set_axis_off()
ax2.set_title('Offset image')

# Calculate the upsampled DFT, again to show what the algorithm is doing
# behind the scenes.  Constants correspond to calculated values in routine.
# See source code for details.
cc_image = _upsampled_dft(image_product, 150, 100, (shift*100)+75).conj()
ax3.imshow(cc_image.real)
ax3.set_axis_off()
ax3.set_title("Supersampled XC sub-area")


plt.show()

print(f"Detected subpixel offset (y, x): {shift}")
#%%
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import numpy.fft
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.gray()  # show the filtered result in grayscale
ascent = misc.ascent()
input_ = numpy.fft.fft2(ascent)
result = ndimage.fourier_shift(input_, shift=200)
result = numpy.fft.ifft2(result)
ax1.imshow(ascent)
ax2.imshow(result.real)  # the imaginary part is an artifact
plt.show()