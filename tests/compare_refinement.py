from scipy.io import loadmat, savemat
from fastatomography.util import *

path = '/home/philipp/projects2/tomo/2019-09-09_kate_pd/05_tomo_with_support/compare/'

meas = loadmat(path+'measured_proj.mat')['m']
ref = loadmat(path+'refined_proj.mat')['r']

m = meas.transpose([2,1,0])
r = ref.transpose([2,1,0])

i = 20
plot(np.abs(r[i]-m[i]))

path = '/home/philipp/projects2/tomo/2019-09-09_kate_pd/05_tomo_with_support/'
supp = np.load(path + 'gaussian_support_nonbinary2x.npy')
savemat(path + 'gaussian_support_nonbinary2x.mat',{'s':supp})

v = np.load(path + '2020-01-13_2x.npy')
savemat(path + '2020-01-13_2x.mat',{'s':v})