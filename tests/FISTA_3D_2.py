
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
from scipy.io import loadmat
from skimage.transform import rotate
import os

os.system("taskset -p 0xff %d" % os.getpid())

path = '/home/philipp/projects/'
path = '/home/philipp/projects2/tomo/2019-04-17-Pd_helix/philipp/'
fn_angles= 'angles_raw.mat'
fn_data = 'data_raw_manual.mat'

angles = loadmat(path+fn_angles)['angles']
angles = angles.T.squeeze()
angles *= -1
angles -= angles.min()

data = loadmat(path+fn_data)['data']
#%%
for i in range(data.shape[2]):
    data[:,:,i] = rotate(data[:,:,i],-90)
    # plot(data[:,:,i])
data = np.transpose(data,[0,2,1])

Horiz_det = Vert_det = N_size = 800
angles_rad = np.deg2rad(angles)
#%%

# for i in range(20):
#     plot(data[:, i, :])

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA method (ASTRA used for projection)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = None, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = None, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-10, # tolerance to stop outer iterations earlier
                    device='gpu')

lc = Rectools.powermethod() # calculate Lipschitz constant
print(f"Lipschitz constant : {lc}")

# Run FISTA reconstrucion algorithm without regularisation
# RecFISTA = Rectools.FISTA(data, iterationsFISTA = 150, lipschitz_const = lc)

# Run FISTA reconstrucion algorithm with 3D regularisation
RecFISTA_reg = Rectools.FISTA(data, iterationsFISTA = 20, \
                              # regularisation = 'FGP_TV', \
                              # regularisation_parameter = 0.002,\
                              # regularisation_iterations = 10,\
                              lipschitz_const = lc)
#%%
sliceSel = 300#int(0.5*N_size)
max_val = RecFISTA_reg.max()
# plt.figure()
# plt.subplot(131)
# plt.imshow(RecFISTA_reg[sliceSel,:,:],vmin=0, vmax=max_val)
# plt.title('3D FISTA regularised reconstruction, axial view')
#
# plt.subplot(132)
# plt.imshow(RecFISTA_reg[:,sliceSel,:],vmin=0, vmax=max_val)
# plt.title('3D FISTA regularised reconstruction, coronal view')

plt.subplots(1,1, figsize=(15,15))
plt.imshow(RecFISTA_reg[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D FISTA regularised reconstruction, sagittal view')
plt.show()


#%%
np.save(path+'RecFISTA_reg.npy',RecFISTA_reg)

