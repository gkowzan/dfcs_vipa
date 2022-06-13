import logging
import numpy as np
from dfcs_vipa import grid
import dfcs_vipa

log = logging.getLogger(__name__)


def grid_overlay(frame, fancy_grid, rio=None):
    """Display a camera frame and a grid on the same image.

    Parameters
    ----------
    frame
        2D Numpy array,
    fancy_grid
        (rows, cols) tuple specifying spectral elements' positions,
    rio
        (row_min, row_max) tuple.

    Returns
    ------
    object
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    grid_map = grid.make_grid_map(*fancy_grid)
    grid_map = np.ma.masked_where(grid_map == False, grid_map)

    rio_stripes = np.zeros((dfcs_vipa.ROWS, dfcs_vipa.COLS), dtype=np.bool)
    if rio is not None:
        rio_stripes[rio[0], :] = True
        rio_stripes[rio[1], :] = True
    rio_stripes = np.ma.masked_where(rio_stripes == False, rio_stripes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(frame)
    ax.imshow(grid_map, cmap=cm.get_cmap('gist_rainbow'), vmax=5000)
    ax.imshow(rio_stripes, cmap=cm.get_cmap('rainbow'), vmax=5000)
    fig.colorbar(im)

    return fig
