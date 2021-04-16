from fastatomography.util import *
from scipy.io import loadmat, savemat
# %%
path = '/home/philipp/projects2/tomo/2019-03-18_Pd_loop/'
# path = '/home/philipp/projects2/tomo/2018-07-03-Pdcoating_few_proj/sample synthesis date 20180615-Pd coating/'
# path = '/home/philipp/projects2/tomo/2019-04-17-Pd_helix/philipp/'
fn = 'reco_blur.npy'
fn = 'thresh_res.mat'
# path = '/home/philipp/projects2/tomo/2019-04-17-Pd_helix/philipp/'
# fn = 'RecFISTA_reg5.npy'
# rec = np.load(path + fn)
#%%
rec = loadmat(path + fn)['r']
# rec = np.transpose(rec, (1,0,2))
# mask = np.load(path + 'mask.npy')
#
# # %%
# blur1 = blur.copy()
# # blur1[:,:60,:] = 0
# # blur1[:,380:,:] = 0
#
# # plot(blur1[:, :, 100])
# # plot(blur1[:, :, 200])
# # plot(blur1[:, :, 300])
# # plot(blur1[:, :, 400])
#
# for i in range(20):
#     plot(blur1[i * 20, :, :])
# # plot(blur1[:, 200, :])
#
#
# # %%
# # plot(blur1[:, 200, :])
# from skimage import io
#
# im = io.imread('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/rec0.tiff')
# print(im.shape)
#
# im = np.transpose(im, (2, 1, 0))
# io.imsave('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/rec0T.tiff', im)
# %%
#
# from skimage import io
#
# im = io.imread('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/reco_blurbin.tiff')
# print(im.shape)
# # %%
# im = np.transpose(im, (1, 2, 0))
# print(im.shape)
# %%
# io.imsave('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/reco_blurbinT.tiff', im)
rec = np.transpose(rec,(1,0,2))
# %%
# mask = (im < 1.1e-16).astype(np.float)
# mask = np.transpose(mask, [2, 1, 0])
# # %%
#
# ms = np.sum(mask, (1, 2))
#
# drawn = ms > 38000
#
# # drawn2 = np.logical_and(np.arange(len(ms))>100,ms > 20000)
#
# # drawn3 = np.logical_or(drawn,drawn2)
#
# f, a = plt.subplots()
# a.plot(np.arange((len(ms))), ms)
# # a.plot(np.arange((len(ms))),drawn3.astype(np.float)*4e4)
# a.plot(np.arange((len(ms))), drawn.astype(np.float) * 3.8e4)
# # a.plot(np.arange((len(ms))),drawn2.astype(np.float)*3e4)
# plt.show()
#
# # %%
# from tqdm import trange
#
# mask2 = mask.copy()
# for i in trange(len(ms)):
#     if not drawn[i]:
#         for j in range(i):
#             if drawn[i - j]:
#                 mask2[i] = mask[i - j]
#                 break

# %%
# for i in trange(100):
#     plot(mask2[i])
# %%
# mask2 = np.transpose(mask2, [2, 1, 0])
# %%
# io.imsave('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/rec0TmaskT2.tiff', mask2)
# %%
# np.save('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/rec0TmaskT2.npy', mask2)

# %%
from collections import deque

import matplotlib.pyplot as plt
import numpy as np


class InteractiveDataPrep(object):
    def __init__(self, data, mask, r=50, action_sequence=None):
        if action_sequence is None:
            action_sequence = [
                (
                    'Now move with the arrow keys and select the position of the same feature again. ENTER', 'enter',
                    'pos',
                    np.ones(3)),
                ('Closing', 'close', 'pos', np.ones(3))
            ]
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 10))
        self.current_action = None
        self.fig = fig
        self.data = data
        self.current_mask = mask
        self.actions = deque(action_sequence)
        fig.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        fig.canvas.mpl_connect('scroll_event', self.scroll)
        fig.canvas.mpl_connect('key_press_event', self.key_press_event)
        fig.canvas.mpl_connect('button_press_event', self.button_press_event)
        fig.canvas.mpl_connect('button_release_event', self.button_release_event)
        self.pos = [0, 0]
        self.ax = ax
        self.r = r
        self.holding_button1 = False
        self.holding_button3 = False
        self.circle1 = plt.Circle((0, 0), self.r, color='r', fill=None)
        ax.add_artist(self.circle1)
        # text location in axes coords
        self.txt = ax.text(0.9, 2, '', transform=ax.transAxes)
        self.data_index = 0

        self.imax = ax.imshow(data[self.data_index], cmap=plt.cm.get_cmap('viridis'))
        self.imax1 = ax1.imshow(self.current_mask[self.data_index], interpolation='nearest', cmap=plt.cm.get_cmap('hot'))
        self.next_action()
        plt.grid(False)
        plt.show()

    def next_action(self):
        self.current_action = self.actions.popleft()
        print(self.current_action[0])
        self.ax.set_title(self.current_action[0])
        self.fig.canvas.draw()
        if self.current_action[1] == 'close':
            self.fig = None
            plt.clf()
            plt.cla()
            plt.close()

    def button_release_event(self, event):
        x, y = int(event.xdata), int(event.ydata)
        self.pos = [y, x]
        #        print self.pos
        print
        event.button
        if self.holding_button1 and event.button == 1:
            self.holding_button1 = False
        elif self.holding_button3 and event.button == 3:
            self.holding_button3 = False

    def refresh_display(self):
        self.imax.set_data(self.data[self.data_index])
        self.imax.set_clim(vmin=self.data[self.data_index].min(), vmax=self.data[self.data_index].max())
        self.imax1.set_data(self.current_mask[self.data_index])
        self.imax1.set_clim(vmin=self.current_mask[self.data_index].min(), vmax=self.current_mask[self.data_index].max())
        plt.draw()

    def button_press_event(self, event):
        x, y = int(event.xdata), int(event.ydata)
        self.pos = [y, x]
        if event.button == 1:
            self.current_mask[self.data_index][
                sector_mask(self.current_mask[self.data_index].shape, self.pos, self.r, (0, 360))] = 1
            self.holding_button1 = True
        elif event.button == 3:
            self.current_mask[self.data_index][
                sector_mask(self.current_mask[self.data_index].shape, self.pos, self.r, (0, 360))] = 0
            self.holding_button3 = True
        # plot(sector_mask(self.current_mask[self.data_index].shape, self.pos, self.r, (0, 360)).astype(np.float))
        self.refresh_display()

    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        self.pos = [y, x]
        # update the line positions
        self.circle1.center = (x, y)
        self.txt.set_text('x=%1.2f, y=%1.2f index=%d' % (self.pos[0], self.pos[1], self.data_index))
        plt.draw()

        if self.holding_button1:
            self.current_mask[self.data_index][
                sector_mask(self.current_mask[self.data_index].shape, self.pos, self.r, (0, 360))] = 1
            self.refresh_display()
        elif self.holding_button3:
            self.current_mask[self.data_index][
                sector_mask(self.current_mask[self.data_index].shape, self.pos, self.r, (0, 360))] = 0
            self.refresh_display()

    def scroll(self, event):
        if not event.inaxes:
            return
        if event.button == 'up':
            self.r += 1
        else:
            self.r -= 1
        x, y = event.xdata, event.ydata
        # update the line positions
        self.circle1.radius = self.r
        plt.draw()

    def key_press_event(self, event):
        #        print(event.key)
        if event.key == 'enter' and self.current_action[1] == 'enter':
            self.current_action[3][:] = [self.data_index, self.pos[0], self.pos[1]]
            self.next_action()
        elif event.key == 'control' and self.current_action[1] == 'control':
            self.current_action[3][:] = [self.r]
            self.next_action()
        elif event.key == 'control' and self.current_action[1] == 'center_radius_control':
            self.current_action[4][:] = [self.r]
            self.current_action[3][:] = [self.pos[0], self.pos[1]]
            self.next_action()
        elif event.key == 'left':
            self.data_index -= 1
            if self.holding_button1:
                self.current_mask[self.data_index][
                    sector_mask(self.current_mask[self.data_index].shape, self.pos, self.r, (0, 360))] = 1
            elif self.holding_button3:
                self.current_mask[self.data_index][
                    sector_mask(self.current_mask[self.data_index].shape, self.pos, self.r, (0, 360))] = 0
            self.txt.set_text('x=%1.2f, y=%1.2f index=%d' % (self.pos[0], self.pos[1], self.data_index))
            self.refresh_display()
        elif event.key == 'right':
            self.data_index += 1
            if self.holding_button1:
                self.current_mask[self.data_index][
                    sector_mask(self.current_mask[self.data_index].shape, self.pos, self.r, (0, 360))] = 1
            elif self.holding_button3:
                self.current_mask[self.data_index][
                    sector_mask(self.current_mask[self.data_index].shape, self.pos, self.r, (0, 360))] = 0
            self.txt.set_text('x=%1.2f, y=%1.2f index=%d' % (self.pos[0], self.pos[1], self.data_index))
            self.refresh_display()


# %%loop_genfire_blur
mask = np.zeros_like(rec)
# mask = loadmat('/home/philipp/projects2/tomo/2019-03-18_Pd_loop/mask_0p7.mat')['m']
# mask = np.transpose(mask,(1,0,2))
d = InteractiveDataPrep(rec, mask, r=50)
mask = np.transpose(mask,(1,0,2))
savemat(path+'thresh_mask.mat',{'d':mask})
# mask = np.transpose(mask,(2,1,0))
# np.save(path+'bin2_threshold_mask.npy',mask)

#%%

# io.imsave(path+'mask.tiff', mask)