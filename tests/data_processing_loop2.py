from fastatomography.util import *
from numpy.fft import fft2, fftshift
from skimage.transform import rotate

path = '/home/philipp/projects2/tomo/2019-03-18_Pd_loop/'
fn = 'Pd helix tomo.emd'
fn2 = 'Pd helix negative tomo.emd'
fn3 = 'tomviz_aligned3.npy'
name = 'loop data'

#%%
proj = np.transpose(np.load(path+fn3),[2,1,0])
for i in range(proj.shape[0]):
    proj[i] = rotate(proj[i],90)
#%%
for i in range(proj.shape[0]):
    plot(proj[i])
#%% equal angle indices

i1 = 40
i2 = 1

ind = [[0,0],[1,1],[2,2],[4,3],[5,4],[7,5],[8,6],[9,7]]

for ii in range(8):

    o1 = ind[ii][0]
    o2 = ind[ii][1]

    print(o1,o2)
    im1 = proj1[i1+o1]
    im2 = proj2[i2+o2]

    print(f"angles: {angles1[i1+o1]} ,  {angles2[i2+o2]}")

    zplot([im1,im2],cmap=['inferno','inferno'],figsize=(18,10))

    im1 = np.log10(1+np.abs(fftshift(fft2(im1))))
    im2 = np.log10(1+np.abs(fftshift(fft2(im2))))

    zplot([im1,im2],cmap=['inferno','inferno'],figsize=(18,10))


#%%
proj = np.transpose(proj1,[2,1,0])
#%%

# plot(np.var(proj[:,:,180:850],0))
#
# proj = proj[:,:800,180:850]
#%%
for i in range(proj.shape[0]):
    plot(proj1[i])

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
plotmosaic(proj2)
#%%

bad = [14,20,31]
plot(proj2[-2])
#%%
proj2 = np.delete(proj2,bad, axis=0)
angles2 = np.delete(angles2,bad, axis=0)
#%%
with h5.File(path+'pd_loop2.h5','w') as f:
    f.create_dataset('stack',data=proj2)
    f.create_dataset('angles', data=angles2)
    np.savetxt(path+'angles2.txt',angles2)
    np.save(path+'stack2.npy',np.transpose(proj2,[2,1,0]))
#%%
bad = [32,42]
plot(proj1[42])
proj1 = np.delete(proj1,bad, axis=0)
angles1 = np.delete(angles1,bad, axis=0)
#%%
proj11 = proj1[:40]
angles11 = angles1[:40]
#%%


with h5.File(path+'pd_loop1.h5','w') as f:
    f.create_dataset('stack',data=proj1)
    f.create_dataset('angles', data=angles1)
    np.savetxt(path+'angles1.txt',angles1)
    np.save(path+'stack1.npy',np.transpose(proj1,[2,1,0]))

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