from fastatomography.default_dependencies import *

path = '/home/philipp/projects2/'
fn = 'Pd5fold_1_step1.mat'
f = loadmat(path + fn)

r = f['atom_pos'].T[:16148]
rc = f['close_pos'].T
v = f['DataMatrix']
# %%

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

ax.scatter(r[:, 0], r[:, 1], r[:, 2], marker='.', s=1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# %%
# fnout = 'Pd5fold.xyz'
# pos_max = r.max(0)
# with open(path + fnout, 'w') as f:
#     f.writelines(f"something something cool\n")
#     f.writelines(f"{pos_max[0]}\t{pos_max[1]}\t{pos_max[2]}\n")
#     for p in r:
#         f.write(
#             "{0:d} {1:.4f} {2:.4f} {3:.4f}\n".format(46, *p[:3])
#         )
#     f.write("-1")
# #%%
# fnout = 'Pd5fold.xtl'
# r /= pos_max
# with open(path + fnout, 'w') as f:
#     f.write("TITLE " + 'self.Title' + "\n CELL \n")
#     f.write("  {0:.2f} {1:.2f} {2:.2f} 90 90 90\n".format(*pos_max))
#     f.write("SYMMETRY NUMBER 1\n SYMMETRY LABEL  P1\n ATOMS \n")
#     f.write("NAME         X           Y           Z" + "\n")
#     for p in r:
#         f.write(
#             "{0} {1:.4f} {2:.4f} {3:.4f}\n".format(
#                 46, *p[:3]
#             )
#         )
#     f.write("EOF")
# #%%
# np.save(path+'Pd5fold.npy', r)
# #%%
# savemat(path+'Pd5fold.mat',{'r': r})
# %%
