import h5py as h5

path = '/home/philipp/projects2/tomo/2019-03-18_Pd_loop/'
fn1 = 'pd_loop1.h5'
fn2 = 'pd_loop2.h5'

name = 'loop data'

with h5.File(path+fn1,'r') as f:
    angles1 = f['angles'][:]
    proj1 = f['stack'][:]


with h5.File(path+fn2,'r') as f:
    angles2 = f['angles'][:]
    proj2 = f['stack'][:]

print(f"proj1.shape {proj1.shape}")
print(f"proj2.shape {proj2.shape}")
print(f"angles1 {angles1}")
print(f"angles2angles1. {angles2}")
