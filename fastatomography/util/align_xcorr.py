import numpy as np


def cross_correlation_align(image, reference, rFilter, kFilter):
    """Align image to reference by cross-correlation"""

    image_f = np.fft.fft2((image - np.mean(image)) * rFilter)
    reference_f = np.fft.fft2((reference - np.mean(reference)) * rFilter)

    xcor = abs(np.fft.ifft2(np.conj(image_f) * reference_f * kFilter))
    shifts = np.unravel_index(xcor.argmax(), xcor.shape)

    # shift image
    output = np.roll(image, shifts[0], axis=0)
    output = np.roll(output, shifts[1], axis=1)

    return shifts, output


def align_xcorr(tilt_series: np.array):
    assert tilt_series

    referenceIndex = tilt_series.shape[0] // 2

    # create Fourier space filter
    filterCutoff = 4
    (Nproj, Ny, Nx) = tilt_series.shape
    ky = np.fft.fftfreq(Ny)
    kx = np.fft.fftfreq(Nx)
    [kX, kY] = np.meshgrid(kx, ky)
    kR = np.sqrt(kX ** 2 + kY ** 2)
    kFilter = (kR <= (0.5 / filterCutoff)) * np.sin(2 * filterCutoff * np.pi * kR) ** 2

    # create real sapce filter to remove edge discontinuities
    y = np.linspace(1, Ny, Ny)
    x = np.linspace(1, Nx, Nx)
    [X, Y] = np.meshgrid(x, y)
    rFilter = (np.sin(np.pi * X / Nx) * np.sin(np.pi * Y / Ny)) ** 2

    offsets = np.zeros((tilt_series.shape[2], 2))

    for i in range(referenceIndex, Nproj - 1):
        offsets[i + 1, :], tilt_series[i + 1, :, :] = cross_correlation_align(
            tilt_series[i + 1, :, :], tilt_series[i, :, :], rFilter, kFilter)

    for i in range(referenceIndex, 0, -1):
        offsets[i - 1, :], tilt_series[i - 1, :, :] = cross_correlation_align(
            tilt_series[i - 1, :, :], tilt_series[i, :, :], rFilter, kFilter)

    # Assign Negative Shifts when Shift > N/2.
    indices_Y = np.where(offsets[:, 0] > tilt_series.shape[1] / 2)
    offsets[indices_Y, 0] -= tilt_series.shape[1]
    indices_X = np.where(offsets[:, 1] > tilt_series.shape[2] / 2)
    offsets[indices_X, 1] -= tilt_series.shape[2]

    return tilt_series, offsets
