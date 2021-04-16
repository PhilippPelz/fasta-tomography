from fastatomography.util import *

path = '/home/philipp/projects2/tomo/2019-04-17-Pd_helix/philipp/'
fn = 'tomviz_aligned2.npy'
name = 'helix data'

proj = np.load(path + fn)

proj = np.transpose(proj,[2,1,0])
#%%

# plot(np.var(proj[:,:,180:850],0))
#
# proj = proj[:,:800,180:850]
#%%
for i in range(proj.shape[0]):
    plot(proj[i], vmax=proj.max())

#%%

#%%
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

# proj1[63] *= 0.9
# proj1[64] *= 0.9
proj_along_rot_axis = np.sum(proj,1)

cm = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=0, vmax=proj_along_rot_axis.shape[0])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

f, a = plt.subplots(figsize=(20,20))
plt.title(f"{name} projection along rotation axis")
for i in range(proj_along_rot_axis.shape[0]):
    colorVal = scalarMap.to_rgba(i)
    a.plot(np.arange(proj_along_rot_axis.shape[1]), proj_along_rot_axis[i], label=f"{i}", color=colorVal)
plt.legend()
plt.show()

#%%

f, a = plt.subplots(figsize=(20,20))
plt.title(f"{name} histogram before truncation")
a.hist(proj.flatten(),200)
plt.show()


proj1 = proj / proj.max()

# proj1[proj1 < -0.05] = 0

proj1 -= proj1.min()
proj1 = proj1 / proj1.max()

f, a = plt.subplots(figsize=(20,20))
plt.title(f"{name} histogram after truncation")
a.hist(proj1.flatten(),200)
plt.show()

#%%
import h5py as h5
plot(proj1[63])
plot(proj1[64])
plot(proj1[65])
fn = '/home/philipp/projects2/tomo/2019-04-17-Pd_helix/Pd helix20190417.emd'
with h5.File(fn,'r') as f:
    angles = f['data']['raw']['tiltangles'][:]

#%%


with h5.File(path+'pd_helix_prep.h5','w') as f:
    f.create_dataset('stack',data=proj1)
    f.create_dataset('angles', data=angles)

#%%
import tomopy
ang = np.deg2rad(angles)
#%%
# Reconstruct object:
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
recon = tomopy.recon(proj, ang, algorithm=tomopy.astra, options=options)



#%%

plot(recon[:,300,:])