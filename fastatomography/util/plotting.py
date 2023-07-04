import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import numpy as np
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.spatial.transform import Rotation as R


def plot_rotating_point(angles, title='title', point=np.array([0, 0, 1]), elevation=55, azimuth=90, savePath=None,
                        dpi=300):
    axis_string = 'YXZ'
    rot = R.from_euler(axis_string,angles.T)
    M = rot.as_matrix()
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    r = point
    r_rot = M @ r

    for rr in r_rot:
        X = [0, rr[0]]
        Y = [0, rr[1]]
        Z = [0, rr[2]]

        line = plt3d.art3d.Line3D(X, Y, Z)
        ax.add_line(line)

    ax.scatter(r_rot[:, 0], r_rot[:, 1], r_rot[:, 2], c='r', marker='x')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elevation, azimuth)
    plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=dpi)


def plot_translations(translations, title='', elevation=90, azimuth=90, savePath=None, dpi=300):
    elevation = elevation
    azimuth = azimuth
    n_angles = translations.shape[0]

    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    for i, rr in enumerate(translations):
        Y = [0, rr[0]]
        X = [0, rr[1]]
        Z = [i, i]

        line = plt3d.art3d.Line3D(X, Y, Z)
        ax.add_line(line)

    ylim = np.max([np.abs(np.min(translations[:, 0])), np.abs(np.max((translations[:, 0])))])
    xlim = np.max([np.abs(np.min(translations[:, 1])), np.abs(np.max((translations[:, 1])))])
    ax.scatter(translations[:, 1], translations[:, 0], np.arange(n_angles), c='r', marker='x')
    ax.set_xlim([-xlim, xlim])
    ax.set_ylim([-ylim, ylim])
    ax.set_zlim([0, n_angles])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('angle number')
    ax.view_init(elevation, azimuth)
    plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=dpi)

def save_stack_gif(save_name, stack, angles, dx, dpi=300, fps=3, duration=None):
    if duration is None:
        duration = stack.shape[0] / fps

    fig, ax = plt.subplots(dpi=dpi)
    title = 'tilt {:-3d}, alpha = {:2.2f}'
    cmap = 'viridis'
    scale = (40 / dx, '4 nm')

    div1 = make_axes_locatable(ax)
    vmax = stack.max()
    imax1 = ax.imshow(stack[0], interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
    ax.set_title(title.format(0, angles[0]))

    for ax1 in [ax]:
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        ax1.grid(False)

    if scale is not None:
        fontprops = fm.FontProperties(size=12)
        scalebar = AnchoredSizeBar(ax.transData,
                                   scale[0], scale[1], 'lower right',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=stack.shape[1] / 40,
                                   fontproperties=fontprops)

        ax1.add_artist(scalebar)
    # plt.tight_layout(pad=0)

    def make_frame(i):
        ind = int(i * fps)
        imax1.set_data(stack[ind])
        ax.set_title(title.format(ind, angles[ind]))
        fig.canvas.draw()
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    animation.write_gif(f'{save_name}.gif', fps=fps)


def save_stack_movie(save_name, stack, angles, dx, dpi=300, fps=10, duration=None, vmin=None, vmax=None, cmap = 'viridis'):
    if duration is None:
        duration = stack.shape[0] / fps

    fig, ax = plt.subplots(dpi=dpi)
    title = 'tilt {:-3d}, alpha = {:2.2f}'
    scale = (40 / dx, '4 nm')
    div1 = make_axes_locatable(ax)
    vmax = vmax if vmax is not None else stack.max()
    vmin = vmin if vmin is not None else stack.min()
    imax1 = ax.imshow(stack[0], interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax, vmin=vmin)
    ax.set_title(title.format(0, angles[0]))

    for ax1 in [ax]:
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        ax1.grid(False)

    if scale is not None:
        fontprops = fm.FontProperties(size=12)
        scalebar = AnchoredSizeBar(ax.transData,
                                   scale[0], scale[1], 'lower right',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=stack.shape[1] / 40,
                                   fontproperties=fontprops)

        ax1.add_artist(scalebar)
    plt.tight_layout(pad=0)

    def make_frame(i):
        ind = int(i * fps)
        imax1.set_data(stack[ind])
        ax.set_title(title.format(ind, angles[ind]))
        fig.canvas.draw()
        return mplfig_to_npimage(fig)

    animation = VideoClip(make_frame, duration=duration)
    animation.write_videofile(f'{save_name}.mp4', fps=fps)



# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(111, projection='3d')
# ax.set_title('Stack histograms')
# nbins = 200
# cmap = matplotlib.cm.get_cmap('viridis')
# for i, s in enumerate(pstack):
#     hist, bins = np.histogram(s, bins=nbins)
#     xs = (bins[:-1] + bins[1:])/2
#     c = cmap(i/pstack.shape[0])
#     ax.bar(xs[1:], hist[1:], zs=i, zdir='y', color=c, ec=c, alpha=0.8)
# # d = np.random.randn(10)
# # ax.scatter(d, d, d, c='r', marker='x')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('angle number')
# # ax.view_init(0, 0)
# plt.show()
# #%%
# fig = plt.figure(dpi=300)
# ax = fig.add_subplot(111)
# ax.set_title('Stack histograms')
# nbins = 200
# cmap = matplotlib.cm.get_cmap('viridis')
# for i, s in enumerate(pstack):
#     hist, bins = np.histogram(s, bins=nbins)
#     xs = (bins[:-1] + bins[1:])/2
#     c = cmap(i/pstack.shape[0])
#     ax.bar(xs[1:], hist[1:], color=c, ec=c, alpha=0.5, label=f'{i}')
# # d = np.random.randn(10)
# # ax.scatter(d, d, d, c='r', marker='x')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# plt.legend()
# # ax.view_init(0, 0)
# plt.show()