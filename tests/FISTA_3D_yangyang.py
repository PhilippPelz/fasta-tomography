# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script to generate 3D analytical phantoms and their projection data with added 
noise and then reconstruct using regularised FISTA algorithm.

Dependencies: 
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CCPi-RGL toolkit (for regularisation), install with 
    conda install ccpi-regulariser -c ccpi -c conda-forge
    or https://github.com/vais-ral/CCPi-Regularisation-Toolkit
    * TomoPhantom, https://github.com/dkazanc/TomoPhantom

@author: Daniil Kazantsev
"""
from fastatomography.util import *
import os

os.system("taskset -p 0xff %d" % os.getpid())

path = '/home/philipp/projects/'
path = '/home/philipp/projects2/tomo/2019-04-17-Pd_helix/philipp/'
# path = '/home/philipp/projects2/tomo/2019-03-18_Pd_loop/'
path = '/home/philipp/projects2/tomo/2018-07-03-Pdcoating_few_proj/sample synthesis date 20180615-Pd coating/'
path = '/home/philipp/projects2/tomo/2019-03-26_Cu_bubble_tomo_yangyang/'
# fn_angles = 'angles.mat'
# fn_data = 'data.mat'

fn_angles = 'angles.txt'
fn_data = 'aligned_cropped_binned.npy'

data = np.load(path + fn_data)

angles = [-78.0,
          -76.0,
          -74.0,
          -72.0,
          -70.0,
          -68.0,
          -66.0,
          -64.0,
          -62.0,
          -60.0,
          -57.0,
          -54.0,
          -51.0,
          -48.0,
          -45.0,
          -42.0,
          -39.0,
          -36.0,
          - 33.0,
          -30.0,
          -27.0,
          -24.0,
          -21.0,
          -18.0,
          - 15.0,
          -12.0,
          -9.0,
          -6.0,
          -3.0,
          0.0,
          3.0,
          6.0,
          9.0,
          12.0,
          15.0,
          18.0,
          21.0,
          24.0,
          27.0,
          30.0,
          33.0,
          36.0,
          39.0,
          42.0,
          45.0,
          48.0,
          51.0,
          54.0,
          57.0,
          60.0,
          62.0,
          64.0,
          66.0,
          68.0,
          70.0,
          72.0,
          74.0,
          76.0,
          78.0
          ]
#
# %%
# data = data['stackCrop2']
# %%
# data = data[100:-100,100:-100,:]
# %%
# vol_mask = np.load(path + 'mask5_3.npy')
# vol_mask = None
# %%
for i in range(5):
    plot(data[:, :, i])
# %%
# for i in range(data.shape[2]):
#     data[:,:,i] = rotate(data[:,:,i],-90)
# plot(data[:,:,i])

data = np.transpose(data, [0, 2, 1])
# vol_mask = np.transpose(vol_mask,[1,0,2])
Horiz_det = data.shape[2]
Vert_det = data.shape[0]
N_size = data.shape[2]
angles_rad = np.deg2rad(angles)

print(f"Horiz_det : {Horiz_det}")
print(f"Vert_det  : {Vert_det}")
print(f"N_size    : {N_size}")
#%%
for i in range(15):
    plot(data[:, i, :])
# %%
gradient_mask = 1  # (data > 0.1).astype(np.float)
data -= data.min()
data /= data.max()
# %%
for i in range(5):
    #     plot(gradient_mask[:, i, :])
    plot(data[:, i, :])

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA method (ASTRA used for projection)")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR
gradient_mask = None
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH=Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                      DetectorsDimV=Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                      CenterRotOffset=None,  # Center of Rotation (CoR) scalar (for 3D case only)
                      AnglesVec=angles_rad,  # array of angles in radians
                      ObjSize=N_size,  # a scalar to define reconstructed object dimensions
                      datafidelity='LS',  # data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                      nonnegativity='ENABLE',  # enable nonnegativity constraint (set to 'ENABLE')
                      OS_number=3,  # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                      tolerance=1e-12,  # tolerance to stop outer iterations earlier
                      device='gpu')

lc = Rectools.powermethod(weights=gradient_mask)  # calculate Lipschitz constant
print(f"Lipschitz constant : {lc}")

# Run FISTA reconstrucion algorithm without regularisation
# RecFISTA = Rectools.FISTA(data, iterationsFISTA = 150, lipschitz_const = lc)
# mask = np.ones(1)
# Run FISTA reconstrucion algorithm with 3D regularisation
vol_mask = None

RecFISTA_reg = Rectools.FISTA(data, gradient_mask, vol_mask, 0.2,
                              weights=gradient_mask,
                              iterationsFISTA=150, \
                              regularisation='TGV', \
                              regularisation_parameter=2e-1, \
                              regularisation_iterations=10, \
                              lipschitz_const=lc)
# %%
# sliceSel = 250  # int(0.5*N_size)
# max_val = RecFISTA_reg.max()
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 15))
# plt.subplot(131)
# ax1.imshow(RecFISTA_reg[sliceSel, :, :], vmin=0, vmax=max_val)
# ax1.set_title('3D FISTA regularised reconstruction, axial view')
# #
# plt.subplot(132)
# ax2.imshow(RecFISTA_reg[:, sliceSel, :], vmin=0, vmax=max_val)
# ax2.set_title('3D FISTA regularised reconstruction, coronal view')
#
# plt.imshow(RecFISTA_reg[:, :, sliceSel], vmin=0, vmax=max_val)
# ax3.title('3D FISTA regularised reconstruction, sagittal view')
# ax3.set_title()

# %%
np.save(path + 'RecFISTA_reg1.npy', RecFISTA_reg)
