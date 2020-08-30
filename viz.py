import matplotlib.pyplot as plt
import numpy as np


# Some helper functions to visualize the results
def _tile_plot(imgs, titles, **kwargs):
    """
    Helper function
    """
    # Create a new figure and plot the three images
    fig, ax = plt.subplots(3, len(imgs[0]), gridspec_kw={'wspace': 0.11, 'hspace': -0.3})
    plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(3):
        for ii, a in enumerate(ax[i]):
            a.set_axis_off()
            a.imshow(imgs[i][ii], interpolation='bilinear', **kwargs)
            if i == 0:
                a.set_title(titles[ii], fontsize=8)
    return fig


def overlay_slices(moving, static, moved, slice_index=None,
                   ltitle='Left', mtitle='Middle', rtitle='Right', fname=None):
    r"""Plot overlaid slices from the given volumes.
    Edited version of dipy.viz.regtools.overlay_slices
    """
    images = []

    sh = static.shape
    moving = np.asarray(moving, dtype=np.float64)
    moving = 255 * (moving - moving.min()) / (moving.max() - moving.min())
    moved = np.asarray(moved, dtype=np.float64)
    moved = 255 * (moved - moved.min()) / (moved.max() - moved.min())
    static = np.asarray(static, dtype=np.float64)
    static = 255 * (static - static.min()) / (static.max() - static.min())

    # Create the color image to draw the overlapped slices into, and extract
    # the slices (note the transpositions)
    if slice_index is None:
        slice_index = sh[0] // 2
    colorll = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
    colorrr = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
    ll = np.asarray(moving[slice_index, :, :]).astype(np.uint8).T
    rr = np.asarray(moved[slice_index, :, :]).astype(np.uint8).T
    mm = np.asarray(static[slice_index, :, :]).astype(np.uint8).T

    colorll[..., 0] = ll * (ll > ll[0, 0])
    colorll[..., 1] = mm * (mm > mm[0, 0])

    colorrr[..., 0] = rr * (rr > rr[0, 0])
    colorrr[..., 1] = mm * (mm > mm[0, 0])

    images.append([ll, colorll, mm, colorrr, rr])

    if slice_index is None:
        slice_index = sh[1] // 2
    colorll = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
    colorrr = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
    ll = np.asarray(moving[:, slice_index, :]).astype(np.uint8).T
    rr = np.asarray(moved[:, slice_index, :]).astype(np.uint8).T
    mm = np.asarray(static[:, slice_index, :]).astype(np.uint8).T

    colorll[..., 0] = ll * (ll > ll[0, 0])
    colorll[..., 1] = mm * (mm > mm[0, 0])

    colorrr[..., 0] = rr * (rr > rr[0, 0])
    colorrr[..., 1] = mm * (mm > mm[0, 0])

    images.append([ll, colorll, mm, colorrr, rr])

    if slice_index is None:
        slice_index = sh[2] // 2
    colorll = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
    colorrr = np.zeros(shape=(sh[2], sh[1], 3), dtype=np.uint8)
    ll = np.asarray(moving[:, :, slice_index]).astype(np.uint8).T
    rr = np.asarray(moved[:, :, slice_index]).astype(np.uint8).T
    mm = np.asarray(static[:, :, slice_index]).astype(np.uint8).T

    colorll[..., 0] = ll * (ll > ll[0, 0])
    colorll[..., 1] = mm * (mm > mm[0, 0])

    colorrr[..., 0] = rr * (rr > rr[0, 0])
    colorrr[..., 1] = mm * (mm > mm[0, 0])

    images.append([ll, colorll, mm, colorrr, rr])

    fig = _tile_plot(images,
                     [ltitle, 'Overlay', mtitle, 'Overlay', rtitle],
                     cmap=plt.cm.gray, origin='lower')

    if fname is not None:
        fig.savefig(fname, bbox_inches='tight', dpi=200)

    return fig
