import numpy as np
import torch as th
import cupy as cp
from numpy.fft import fft2, ifft2, fftn, ifftn, fftshift, ifftshift, fftfreq, rfftfreq
from scipy.io import loadmat, savemat
from numpy.linalg import norm, svd, eig, eigh
import matplotlib.pyplot as plt
from tqdm import trange
import psutil
import GPUtil
import os
import copy
import logging


def fourier_coordinates_2D(N, dx=[1.0, 1.0], centered=True):
    qxx = fftfreq(N[1], dx[1])
    qyy = fftfreq(N[0], dx[0])
    if centered:
        qxx += 0.5 / N[1] / dx[1]
        qyy += 0.5 / N[0] / dx[0]
    qx, qy = np.meshgrid(qxx, qyy)
    q = np.array([qy, qx]).astype(np.float32)
    return q
