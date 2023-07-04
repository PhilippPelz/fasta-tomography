import numpy as np
import torch as th
from scipy.io import loadmat
import warnings
import h5py

def load(fn, sub, to_tensor=False):                              
    if fn.endswith('.mat'):
        out = np.deg2rad(loadmat(fn)[sub])
    elif fn.endswith('.h5'):
        with h5py.File( fn, 'r') as f:
            out = np.squeeze(f[sub][...])
    else:
        raise Exception("Expected file {} to be a matlab file with endian '.mat' or h5 file with endian '.h5'. Please input a matlab file, h5 file or edit the load() method in han_utils so that it will read-in other formats.".format(fn))
    if to_tensor:
        out = th.as_tensor(out).squeeze().contiguous()
    return out


class Angles():
    def __init__(self, path_to_angles_file, matlab_array_name):
        self.path = path_to_angles_file
        self.matlab_name = matlab_array_name
        
        angles_in     = self.load()
        self.angles   = self.correct_dimensions(angles_in)
        self.n_angles = self.angles.shape[1]

    def load(self):
        if self.path.endswith('.mat'):
            angles_in = np.deg2rad(loadmat(self.path)[self.matlab_name])
        elif self.path.endswith('.h5'):
            with h5py.File( self.path, 'r') as f:
                angles_in = np.squeeze(f[self.matlab_name][...])
        else:
            raise Exception("Expected angle file (angle_fn) to be a matlab file with endian '.mat', or h5 file with endian .h5. Please input a matlab file or edit the load() method of the class Angles so that it will read in other formats.")
        return th.as_tensor(angles_in.T).squeeze().contiguous()

    def correct_dimensions(self,angles_in):
        n_angles = max(angles_in.shape)
        if n_angles <= 3:
            warnings.warn("Number of projection angles was determined to be <= 3. This suggests that you are either attempting to reconstruct with just 3 projections or the angle file wasn't interpreted correctly.")
            
        angles = np.zeros((3, n_angles), dtype=np.float32)
        if angles_in.ndim == 1:
            angles[0, :] = 0
            angles[1, :] = angles_in
            angles[2, :] = 0
        elif angles_in.ndim == 2:
            max_dim = np.where(angles_in=max(angles_in.shape))               # Get the longest dimension
            angles_in = np.transpose(angles_in, (abs(1 - max_dim), max_dim) ) # Set the longest dimension to be the rows
            angles = angles_in
        else:
            raise Exception("Angles not correct dimension. Expected angle array with 1 or 2 dimensions.")
        return angles
