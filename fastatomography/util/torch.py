import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from numpy.fft import fft2, fftn, ifft2, ifftn, fftshift, ifftshift, fft, ifft
import torch as th
import torch.distributed as dist
from cmath import phase
import cmath as cm
import math as m
from math import cos, sin, sqrt
import numba.cuda as cuda
from .torch_complex import *

re = np.s_[..., 0]
im = np.s_[..., 1]

def memory_mb(x, dtype=None):
    if isinstance(x, th.Tensor):
        return x.nelement() * x.element_size() / 2 ** 20
    elif isinstance(x, tuple):
        assert dtype is not None, 'memory_mb: dtype must not be None'
        element_size = th.zeros(1, dtype=dtype).element_size()
        nelement = np.prod(np.asarray(x))
        return nelement * element_size / 2 ** 20


def memory_gb(x, dtype=None):
    if isinstance(x, th.Tensor):
        return x.nelement() * x.element_size() / 2 ** 30
    elif isinstance(x, tuple):
        assert dtype is not None, 'memory_mb: dtype must not be None'
        element_size = th.zeros(1, dtype=dtype).element_size()
        nelement = np.prod(np.asarray(x))
        return nelement * element_size / 2 ** 30

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return th.cat([back, front], axis)


def circshift(x, axes, shifts):
    real, imag = th.unbind(x, -1)
    for ax, sh in zip(axes, shifts):
        real = roll_n(real, axis=ax, n=sh)
        imag = roll_n(imag, axis=ax, n=sh)
    return th.stack((real, imag), -1)


def memory_mb(x, dtype=None):
    if isinstance(x, th.Tensor):
        return x.nelement() * x.element_size() / 2 ** 20
    elif isinstance(x, tuple):
        assert dtype is not None, 'memory_mb: dtype must not be None'
        element_size = th.zeros(1, dtype=dtype).element_size()
        nelement = np.prod(np.asarray(x))
        return nelement * element_size / 2 ** 20


def memory_gb(x, dtype=None):
    if isinstance(x, th.Tensor):
        return x.nelement() * x.element_size() / 2 ** 30
    elif isinstance(x, tuple):
        assert dtype is not None, 'memory_mb: dtype must not be None'
        element_size = th.zeros(1, dtype=dtype).element_size()
        nelement = np.prod(np.asarray(x))
        return nelement * element_size / 2 ** 30


@cuda.jit
def prox_D_gaussian_kernel(z, z_hat, a, beta, a_strides):
    """

    :param z:           D x K x My x Mx x 2
    :param z_hat:       D x K x My x Mx x 2
    :param a:           D x K x My x Mx
    :param beta:        1
    :param a_strides:   (4,)
    :return:
    """
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    D, K, MY, MX, _ = z.shape
    N1 = D * K * MY * MX

    d = n // a_strides[0]
    k = (n - d * a_strides[0]) // a_strides[1]
    my = (n - d * a_strides[0] - k * a_strides[1]) // a_strides[2]
    mx = (n - d * a_strides[0] - k * a_strides[1] - my * a_strides[2]) // a_strides[3]

    if n < N1:
        z_hatc = z_hat[d, k, my, mx, 0] + 1j * z_hat[d, k, my, mx, 1]
        abs_zhat_c = abs(z_hatc)
        if abs_zhat_c != 0:
            sgn_zhat = z_hatc / abs_zhat_c
            fac = (beta * a[d, k, my, mx] + abs_zhat_c) / (1.0 + beta)
            zc = fac * sgn_zhat
            z[d, k, my, mx, 0] = zc.real
            z[d, k, my, mx, 1] = zc.imag


def sgn(x):
    return x / complex_abs(x).view((*x.shape[:-1], 1)).expand_as(x)


def R_factor(z, a, world_size=1):
    num = norm1((complex_abs(z) - a).flatten())
    denom = norm1(a.flatten())
    if world_size > 1:
        dist.all_reduce(num)
        dist.all_reduce(denom)
    return num / denom

def prox_D_gaussian(z, z_hat, a, beta):
    """

    :param z:           D x K x My x Mx x 2
    :param z_hat:       D x K x My x Mx x 2
    :param a:           D x K x My x Mx
    :param beta:        1
    :param a_strides:   (4,)
    :return:
    """
    gpu = cuda.get_current_device()
    threadsperblock = gpu.MAX_THREADS_PER_BLOCK // 2
    blockspergrid = m.ceil(np.prod(np.array(a.shape)) / threadsperblock)
    strides = th.tensor(a.stride()).to(z.device)

    prox_D_gaussian_kernel[blockspergrid, threadsperblock](z, z_hat, a, beta, strides)


def vdot(a: th.Tensor, b: th.Tensor):
    """
    Return the dot product of two vectors. The vdot(a, b) function handles complex numbers differently than dot(a, b).
    If the first argument is complex the complex conjugate of the first argument is used for the calculation
    of the dot product.
    :param a: tensor
    :param b: tensor
    :return: tensor
    """
    if a.shape[-1] != 2 or b.shape[-1] != 2:
        raise RuntimeWarning('taking vdot of non-complex tensor! a.shape ' + str(a.shape) + 'b.shape ' + str(b.shape))
    are = a[re]
    aim = -a[im]
    bre = b[re]
    bim = b[im]
    x = are * bre - aim * bim
    real_res = th.sum(x).item()
    x = are * bim + aim * bre
    imag_res = th.sum(x).item()
    return real_res + 1j * imag_res


# %%
def norm(x, axis=None):
    """
    Frobenius vector norm
    :param x: tensor
    :return: Norm of the vector
    """
    if axis is not None:
        s = th.sum(complex_abs2(x), axis)
    else:
        s = th.sum(complex_abs2(x))
    res = th.sqrt(s)
    return res


def norm1(x, axis=None):
    """
    1 vector norm
    :param x: tensor
    :return: Norm of the vector
    """
    if axis is not None:
        s = th.sum(th.abs(x), axis)
    else:
        s = th.sum(th.abs(x))
    return s


def distance(z, x):
    """
    Distance of two complex vectors
    :param z: tensor
    :param x: tensor
    :return:
    """
    c = vdot(z, x)
    phi = -phase(c)
    exp_minus_phi = cos(phi) + 1j * sin(phi)
    p = cx_from_numpy(np.array([exp_minus_phi]))
    p = p.to(x.device)
    x_hat = complex_mul(x, p)
    res = norm(z - x_hat)
    return res


def rel_dist(z, x):
    """
    Distance of two complex vectors
    :param z: tensor
    :param x: tensor
    :return:
    """
    d = distance(z, x)
    x_norm = norm(x)
    return d / x_norm


# def numpy_rel_dist(z, x):
#     c = np.vdot(z, x)
#     phi = -phase(c)
#     exp_minus_phi = cos(phi) + 1j * sin(phi)
#     x_hat = x * exp_minus_phi
#     dist = np.linalg.norm(z - x_hat)
#     x_norm = np.linalg.norm(x)
#     return dist / x_norm

# %%
# x = np.random.randn(5) + 1j * np.random.randn(5)
# y = np.random.randn(5) + 1j * np.random.randn(5)
#
# np_res = numpy_rel_dist(x, y)
#
# x1 = cx_from_numpy(x)
# y1 = cx_from_numpy(y)
# # %%
# th_res = rel_dist(x1, y1)
#
# print(f"numpy result: {np_res}")
# print(f"my    result: {th_res}")
def conj(a: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2:
        raise RuntimeWarning('taking conj of non-complex tensor!')
    real = a[re] * 1
    imag = -1 * a[im]
    return th.stack([real, imag], -1)


def complex_abs(a: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2:
        raise RuntimeWarning('taking complex_abs of non-complex tensor!')
    return th.sqrt(a[re] ** 2 + a[im] ** 2)


def complex_abs2(a: th.Tensor) -> th.Tensor:
    if a.shape[-1] != 2:
        raise RuntimeWarning('taking complex_abs2 of non-complex tensor!')
    return a[re] ** 2 + a[im] ** 2


def cangle(a: th.Tensor, deg=0) -> th.Tensor:
    if a.shape[-1] != 2:
        raise RuntimeWarning('taking exp of non-complex tensor!')
    if deg:
        fact = 180 / np.pi
    else:
        fact = 1.0
    return th.atan2(a[im], a[re]) * fact


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return th.cat([back, front], axis)


def fftshift2(x):
    real, imag = th.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim) // 2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return th.stack((real, imag), -1)  # last dim=2 (real&imag)


def ifftshift2(x):
    real, imag = th.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return th.stack((real, imag), -1)  # last dim=2 (real&imag)


def focused_probe(E, N, d, alpha_rad, defocus_nm, det_pix=40e-6, C3_um=1000, C5_mm=1, tx=0, ty=0, Nedge=15,
                  aperture=True, plot=False):
    emass = 510.99906;  # electron rest mass in keV
    hc = 12.3984244;  # h*c
    lam = hc / np.sqrt(E * 1e-3 * (2 * emass + E * 1e-3))  # in Angstrom
    #    print 'lambda : %g' % lam
    alpha = alpha_rad
    tilt_x = 0
    tilt_y = 0

    phi = {
        '11': 0,
        '22': 0,
        '33': 0.0,
        '44': 0.0,
        '55': 0.0,
        '66': 0.0,
        '31': 0.0,
        '42': 0.0,
        '53': 0.0,
        '64': 0.0,
        '51': 0.0,
        '62': 0.0
    }

    a_dtype = {'names': ['10', '11', '20', '22',
                         '31', '33',
                         '40', '42', '44',
                         '51', '53', '55',
                         '60', '62', '64', '66'],
               'formats': ['f8'] * 16}
    a0 = np.zeros(1, a_dtype)
    a0['10'] = 0
    a0['11'] = 0
    a0['20'] = defocus_nm  # defocus: -60 nm
    a0['22'] = 0
    a0['31'] = 0
    a0['33'] = 0
    a0['22'] = 0
    a0['40'] = C3_um  # C3/spherical aberration: 1000 um
    a0['42'] = 0
    a0['44'] = 0
    a0['60'] = C5_mm  # C5/Chromatic aberration: 1 mm

    dk = 1.0 / (N * d)
    qmax = np.sin(alpha) / lam
    ktm = np.arcsin(qmax * lam)
    detkmax = np.arcsin(lam / (2 * d))
    d_alpha = detkmax / (N / 2)

    z = det_pix * N * d / lam

    print('alpha         [mrad]     = %g' % (alpha * 1000))
    print('alpha_max det [mrad]     = %g' % (detkmax * 1000))

    print('qmax                     = %g' % qmax)
    print('beam  dmin [Angstrom]    = %g' % (1 / qmax / 2))
    print('dkmax                    = %g' % (dk * N / 2))
    print('detec dmin [Angstrom]    = %g' % (1 / (dk * N / 2) / 2))
    print('z                 [m]    = %g' % z)

    scalekmax = d_alpha * 50
    # print 'scale bar     [mrad]     = %g' % (scalekmax*1000)

    kx, ky = np.meshgrid(dk * (-N / 2. + np.arange(N)) + tilt_x, dk *
                         (-N / 2. + np.arange(N)) + tilt_y)

    k2 = np.sqrt(kx ** 2 + ky ** 2)
    # riplot(k2,'k2')
    ktheta = np.arcsin(k2 * lam)
    kphi = np.arctan2(ky, kx)
    # riplot(ktheta,'ktheta')
    scaled_a = a0.copy().view(np.float64)
    scales = np.array([10, 10,  # a_2x, nm->A
                       10, 10,  # a_2x, nm->A
                       10, 10,  # a_3x, nm->A
                       1E4, 1E4, 1E4,  # a_4x, um->A
                       1E4, 1E4, 1E4,  # a_5x, um->A
                       1E7, 1E7, 1E7, 1E7,  # a_6x, mm->A
                       ], dtype=np.float64)
    scaled_a *= scales
    a = scaled_a.view(a_dtype)

    cos = np.cos
    chi = 2.0 * np.pi / lam * (1.0 * (a['11'] * cos(2 * (kphi - phi['11'])) + a['10']) * ktheta + 1.0 / 2 * (
            a['22'] * cos(2 * (kphi - phi['22'])) + a['20']) * ktheta ** 2 +
                               1.0 / 3 * (a['33'] * cos(3 * (kphi - phi['33'])) + a['31'] * cos(
                1 * (kphi - phi['31']))) * ktheta ** 3 +
                               1.0 / 4 * (a['44'] * cos(4 * (kphi - phi['44'])) + a['42'] * cos(
                2 * (kphi - phi['42'])) + a['40']) * ktheta ** 4 +
                               1.0 / 5 * (a['55'] * cos(5 * (kphi - phi['55'])) + a['53'] * cos(
                3 * (kphi - phi['53'])) + a['51'] * cos(1 * (kphi - phi['51']))) * ktheta ** 5 +
                               1.0 / 6 * (a['66'] * cos(6 * (kphi - phi['66'])) + a['64'] * cos(
                4 * (kphi - phi['64'])) + a['62'] * cos(2 * (kphi - phi['62'])) + a['60']) * ktheta ** 6)
    # riplot(chi,'chi')
    if aperture:
        arr = np.zeros((N, N), dtype=np.complex64)
    else:
        arr = np.ones((N, N), dtype=np.complex64)
    arr[ktheta < ktm] = 1
    # riplot(arr,'arr')
    dEdge = Nedge / (qmax / dk);  # fraction of aperture radius that will be smoothed
    # some fancy indexing: pull out array elements that are within
    #    our smoothing edges
    ind = np.bitwise_and((ktheta / ktm > (1 - dEdge)),
                         (ktheta / ktm < (1 + dEdge)))
    arr[ind] = 0.5 * (1 - np.sin(np.pi / (2 * dEdge) * (ktheta[ind] / ktm - 1)))
    arr *= np.exp(1j * chi);

    arr = fftshift(arr)

    arr_real = fftshift(ifft2(arr))
    arr_real /= np.linalg.norm(arr_real)
    #    arr_real = nd.zoom(arr_real.real,0.5) + 1j * nd.zoom(arr_real.imag,0.5)

    return np.real(arr_real).astype(np.float32), np.imag(arr_real).astype(np.float32), np.real(arr).astype(
        np.float32), np.imag(arr).astype(np.float32)


def plot_abs_phase_mosaic(img, suptitle='Image', savePath=None, cmap=['bone', 'bone'], title=['Abs', 'Phase'],
                          show=True):
    plotzmosaic([(np.log10(np.abs(img))), np.angle(img)], suptitle, savePath, cmap, title, show)


def plot_abs_phase_mosaic2(img, suptitle='Image', savePath=None, cmap=['bone', 'bone'], title=['Abs', 'Phase'],
                           show=True):
    plotzmosaic([(np.abs(img)), np.angle(img)], suptitle, savePath, cmap, title, show)


def plot_re_im_mosaic(img, suptitle='Image', savePath=None, cmap=['viridis', 'viridis'], title=['Real', 'Imag'],
                      show=True):
    plotzmosaic([np.real(img), np.imag(img)], suptitle, savePath, cmap, title, show)


def plotzmosaic(img, suptitle='Image', savePath=None, cmap=['hot', 'hsv'], title=['Abs', 'Phase'], show=True):
    im1, im2 = img
    mos1 = mosaic(im1)
    mos2 = mosaic(im2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    div1 = make_axes_locatable(ax1)
    div2 = make_axes_locatable(ax2)
    fig.suptitle(suptitle, fontsize=20)
    imax1 = ax1.imshow(mos1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))
    imax2 = ax2.imshow(mos2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[1]))
    cax1 = div1.append_axes("right", size="10%", pad=0.05)
    cax2 = div2.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(imax1, cax=cax1)
    cbar2 = plt.colorbar(imax2, cax=cax2)
    ax1.set_title(title[0])
    ax2.set_title(title[1])
    plt.tight_layout()
    ax1.grid(False)
    ax2.grid(False)
    if show:
        plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=600)


def plotcxmosaic(img, title='Image', savePath=None, cmap='hot', show=True, figsize=(10, 10), vmax=None):
    fig, ax = plt.subplots(figsize=figsize)
    mos = imsave(mosaic(img))
    cax = ax.imshow(mos, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
    ax.set_title(title)
    plt.grid(False)
    plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=600)


def HSV_to_RGB(cin):
    """\
    HSV to RGB transformation.
    """

    # HSV channels
    h, s, v = cin

    i = (6. * h).astype(int)
    f = (6. * h) - i
    p = v * (1. - s)
    q = v * (1. - s * f)
    t = v * (1. - s * (1. - f))
    i0 = (i % 6 == 0)
    i1 = (i == 1)
    i2 = (i == 2)
    i3 = (i == 3)
    i4 = (i == 4)
    i5 = (i == 5)

    imout = np.zeros(h.shape + (3,), dtype=h.dtype)
    imout[:, :, 0] = 255 * (i0 * v + i1 * q + i2 * p + i3 * p + i4 * t + i5 * v)
    imout[:, :, 1] = 255 * (i0 * t + i1 * v + i2 * v + i3 * q + i4 * p + i5 * p)
    imout[:, :, 2] = 255 * (i0 * p + i1 * p + i2 * t + i3 * v + i4 * v + i5 * q)

    return imout


def P1A_to_HSV(cin, vmin=None, vmax=None):
    """\
    Transform a complex array into an RGB image,
    mapping phase to hue, amplitude to value and
    keeping maximum saturation.
    """
    # HSV channels
    h = .5 * np.angle(cin) / np.pi + .5
    s = np.ones(cin.shape)

    v = abs(cin)
    if vmin is None: vmin = 0.
    if vmax is None: vmax = v.max()
    assert vmin < vmax
    v = (v.clip(vmin, vmax) - vmin) / (vmax - vmin)

    return HSV_to_RGB((h, s, v))


def imsave(a, filename=None, vmin=None, vmax=None, cmap=None):
    """
    imsave(a) converts array a into, and returns a PIL image
    imsave(a, filename) returns the image and also saves it to filename
    imsave(a, ..., vmin=vmin, vmax=vmax) clips the array to values between vmin and vmax.
    imsave(a, ..., cmap=cmap) uses a matplotlib colormap.
    """

    if a.dtype.kind == 'c':
        # Image is complex
        if cmap is not None:
            print('imsave: Ignoring provided cmap - input array is complex')
        i = P1A_to_HSV(a, vmin, vmax)
        im = Image.fromarray(np.uint8(i), mode='RGB')

    else:
        if vmin is None:
            vmin = a.min()
        if vmax is None:
            vmax = a.max()
        im = Image.fromarray((255 * (a.clip(vmin, vmax) - vmin) / (vmax - vmin)).astype('uint8'))
        if cmap is not None:
            r = im.point(lambda x: cmap(x / 255.0)[0] * 255)
            g = im.point(lambda x: cmap(x / 255.0)[1] * 255)
            b = im.point(lambda x: cmap(x / 255.0)[2] * 255)
            im = Image.merge("RGB", (r, g, b))
            # b = (255*(a.clip(vmin,vmax)-vmin)/(vmax-vmin)).astype('uint8')
            # im = Image.fromstring('L', a.shape[-1::-1], b.tostring())

    if filename is not None:
        im.save(filename)
    return im


def mosaic(data):
    n, w, h = data.shape
    diff = np.sqrt(n) - int(np.sqrt(n))
    s = np.sqrt(n)
    m = int(s)
    if diff > 1e-6: m += 1
    mosaic = np.zeros((m * w, m * h)).astype(data.dtype)
    for i in range(m):
        for j in range(m):
            if (i * m + j) < n:
                mosaic[i * w:(i + 1) * w, j * h:(j + 1) * h] = data[i * m + j]
    return mosaic


def plotmosaic(img, title='Image', savePath=None, cmap='hot', show=True, figsize=(10, 10), vmax=None):
    fig, ax = plt.subplots(figsize=figsize)
    mos = mosaic(img)
    cax = ax.imshow(mos, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
    cbar = fig.colorbar(cax)
    ax.set_title(title)
    plt.grid(False)
    plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=600)


def zplot(imgs, suptitle='Image', savePath=None, cmap=['hot', 'hsv'], title=['Abs', 'Phase'], show=True,
          figsize=(9, 5), scale=None):
    im1, im2 = imgs
    fig = plt.figure(figsize=figsize)
    fig.suptitle(suptitle, fontsize=15, y=0.8)
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(wspace=0, hspace=0)  # set the spacing between axes.
    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs1[1])
    div1 = make_axes_locatable(ax1)
    div2 = make_axes_locatable(ax2)

    imax1 = ax1.imshow(im1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))
    imax2 = ax2.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[1]))

    cax1 = div1.append_axes("left", size="10%", pad=0.4)
    cax2 = div2.append_axes("right", size="10%", pad=0.4)

    cbar1 = plt.colorbar(imax1, cax=cax1)
    cbar2 = plt.colorbar(imax2, cax=cax2)

    cax1.yaxis.set_ticks_position('left')
    ax2.yaxis.set_ticks_position('right')

    ax1.set_title(title[0])
    ax2.set_title(title[1])

    if scale is not None:
        fontprops = fm.FontProperties(size=18)
        scalebar = AnchoredSizeBar(ax1.transData,
                                   scale[0], scale[1], 'lower right',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=im1.shape[0] / 40,
                                   fontproperties=fontprops)

        ax1.add_artist(scalebar)

    ax1.grid(False)
    ax2.grid(False)

    if show:
        plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=600)
