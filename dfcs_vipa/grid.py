import logging
import itertools as it
import numpy as np
from dfcs_vipa.experiment import find_maxima, find_index
from dfcs_vipa.units import nu2wn, nu2lambda, lambda2nu
import dfcs_vipa

log = logging.getLogger(__name__)


def get_rio_pos(arr):
    """Find coordinates of CW peaks from camera array.

    Parameters
    ----------
    arr : ndarray
        2D array with camera image containing only the first- and
        second-order diffracted CW laser.

    Returns
    -------
    rows : ndarray
    cols : ndarray
        Rows and columns specifying the coordinates of the CW peak in
        `arr`.  First elements of `rows` and `cols` correspond
        to the CW peak with lower row value, i.e. the peak which is
        higher in the camera image.
    """
    half_row = dfcs_vipa.ROWS//2
    row_min, row_max = np.argmax(arr[:half_row, :]), np.argmax(arr[half_row:, :])
    rows, cols = np.unravel_index(
        (row_min, row_max), arr.shape
    )
    rows -= 1; rows[1] += half_row

    return rows, cols


def limit_grid(points, rows, cols=(0, dfcs_vipa.COLS)):
    """Removes grid points outside the rectangle defined by `rows` and
    `cols`.

    Used in conjunction with comb tooth mask to remove the grid points
    whose comb teeth fall (partially) outside of the camera array and
    whose intensities cannot be retrieved.

    Parameters
    ----------
    points: list of tuple
        (row, col) tuples defining the VIPA spectrometer diffraction
        pattern,
    rows: tuple
        (row_min, row_max) of the row numbers delimiting
        allowable region of the camera,
    cols: tuple
        (col_min, col_max) of the column numbers delimiting
        allowable region of the camera.

    Returns
    -------
    list of tuples
        (row, col) tuples defining the VIPA spectrometer diffraction
        pattern, without the points falling outside the permitted
        region.
    """
    return [p for p in points
            if p[0] > rows[0] and p[0] < rows[1]
            and p[1] > cols[0] and p[1] < cols[1]]


def limit_grid_trapz(points, rows, cols=(0, dfcs_vipa.COLS)):
    """Removes grid points outside the trapezoid defined by `rows` and `cols`.

    Parameters
    ----------
    points: list of tuple
        (row, col) tuples defining the VIPA spectrometer diffraction
        pattern,
    rows: tuple of tuple
        ((u_min, u_max), (l_min, l_max)) - row numbers defining the
        trapezoid vertices; the trapezoid is assumed to be limited by
        leftmost and rightmost columns;
    cols: tuple
        (col_min, col_max) - column numbers delimiting allowable region
        of the camera.

    Returns
    -------
    list of tuples
        (row, col) tuples defining the VIPA spectrometer diffraction
        pattern, without the points falling outside the permitted
        region.
    """
    (u_min, u_max), (l_min, l_max) = rows

    def row_max(col):
        intercept = u_min
        slope = (u_max-u_min)/dfcs_vipa.COLS

        return np.round(intercept + slope*col).astype(np.int)

    def row_min(col):
        intercept = l_min
        slope = (l_max-l_min)/dfcs_vipa.COLS

        return np.round(intercept - slope*col).astype(np.int)

    return [p for p in points
            if p[0] > row_min(p[1]) and p[0] < row_max(p[1])
            and p[1] > cols[0] and p[1] < cols[1]]


def collect_column(arr, start_row, start_col, row_min, row_max):
    """Follow the VIPA stripe and return coords lying on it.

    The tracking procedure starts at (`start_row`, `start_col`) and goes
    to the top and the the bottom of the stripe.  At each step the row
    number is incremented (decremented) and the brightest point in
    `arr[row, col-rad:col+rad+1]` horizontal stripe is taken as the new
    starting point and saved to the `points` list.

    Parameters
    ----------
    arr : ndarray
        2D camera array containing relatively uniformly illuminated VIPA
        diffraction image of unresolved comb modes, consisting of almost
        vertical stripes corresponding to different diffraction orders
        of the etalon.
    start_row : int
        row of the image from the which the stripe will be tracked to
        the bottom and to the top of the array.
    start_col: int
        the column at which the intensity of the stripe is at the
        maximum.

    Returns
    -------
    points : list of tuple
        An unsorted list of (row, col) tuples, which are the coordinates
        of the brightest points lying on the stripe.
    """
    rad = 1
    points = []

    col = start_col
    for row in range(start_row, row_min-1, -1):
        left_lim = col-rad if col-rad>0 else 0
        right_lim = col+rad+1 if col+rad+1 < dfcs_vipa.COLS else dfcs_vipa.COLS
        strip = arr[row, left_lim:right_lim]
        col = col-rad + np.argmax(strip)
        if col < 0:
            col = 0
        elif col >= dfcs_vipa.COLS:
            col = dfcs_vipa.COLS - 1
        points.append((row, col))

    col = start_col
    for row in range(start_row+1, row_max):
        left_lim = col-rad if col-rad>0 else 0
        right_lim = col+rad+1 if col+rad+1 < dfcs_vipa.COLS else dfcs_vipa.COLS
        strip = arr[row, left_lim:right_lim]
        col = col-rad + np.argmax(strip)
        if col < 0:
            col = 0
        elif col >= dfcs_vipa.COLS:
            col = dfcs_vipa.COLS - 1
        points.append((row, col))

    return points


def make_grid(arr, rio_rows=np.array([0, dfcs_vipa.ROWS])):
    """Find coordinates lying on each VIPA diffraction order.

    The resultant grid of points is used in further data analysis to
    retrieve the intensities of unresolved comb modes or, after further
    filtering, resolved comb modes.  The frequency of the spectral elements
    increases when going from higher row numbers to lower row numbers (from
    bottom to top) along the stripe and going from lower to higher column
    number (from left to right).  The coordinates are sorted accordingly
    before being returned from this function.

    Parameters
    ----------
    arr : ndarray
        2D camera array containing relatively uniform VIPA diffraction
        image of unresolved comb modes, consisting of almost vertical
        stripes corresponding to different diffraction orders of the
        etalon.
    start_row : int
        row of the image from the which the stripe will be tracked to
        the bottom and to the top of the array.

    Returns
    -------
    grid_points : list of tuple
        A sorted list of (row, col) tuples, which are the coordinates of
        the points lying on the stripes (diffraction orders) comprising
        the VIPA spectrometer diffraction pattern.
    """
    # threshold = np.mean(arr[-1, :])
    row_min, row_max = np.min(rio_rows), np.max(rio_rows)
    start_row = (row_max+row_min)//2
    columns = find_maxima(arr[start_row, :], window_len=1, thres=0, order=2)
    grid_points = []

    for start_col in columns:
        points = collect_column(arr, start_row, start_col, row_min, row_max)
        points.sort(key=lambda x: dfcs_vipa.ROWS-1-x[0])
        grid_points.extend(points)

    # return the points ascending in frequency
    return grid_points[::-1]


def grid2fancy(grid):
    """Convert list of tuples to a tuple of rows and cols arrays.

    Used for fancy indexing of NumPy array.

    Parameters
    ----------
    grid : list of tuples
        A sorted list of (row, col) tuples, which are the coordinates of
        the points lying on the stripes (diffraction orders) comprising
        the VIPA spectrometer diffraction pattern.

    Returns
    -------
    rows : ndarray
        First elements of tuples in `grid`.
    cols : ndarray
        Second elements of tuples in `grid`.
    """
    rows = np.array(
        [p[0] for p in grid]
    )
    cols = np.array(
        [p[1] for p in grid]
    )

    return rows, cols


def fancy2grid(fancy):
    """Convert a tuple or rows and cols arrays to a list of tuples."""
    return [(r, c) for r, c in zip(fancy[0], fancy[1])]


def identify_tooth(small, large):
    """Return index of first tooth from `small` grid in `large` grid

    Parameters
    ----------
    small : tuple of ndarray
        Fancy indexing tuple containing comb teeth positions.
    large : tuple of ndarray
        Fancy indexing tuple containing comb teeth positions.

    Returns
    -------
    int
    """
    small, large = fancy2grid(small), fancy2grid(large)

    return large.index(small[0])


def closest_to_rio(rio, fancy_grid):
    """Return the grid index of the tooth closest to RIO position."""
    rows, cols = fancy_grid
    selector = np.where(cols == rio[1])[0]
    rows_limited = rows[selector]
    i_tooth = find_index(rows_limited, rio[0])

    return selector[i_tooth]


def coerce_teeth_grid(ref, adj):
    """Remove teeth from `adj` which do not correspond to the ones from `ref`.

    The algorithm goes over every element in `ref`, looks for the closest
    element in `adj` within the same column and adds it to a new grid.
    Effectively filters out spurious teeth from `adj`.

    Parameters
    ----------
    ref : list of tuple
        teeth grid after limiting by RIO rows,
    adj: list of tuple
        full teeth grid to be filtered.

    Returns
    -------
    list of tuple
        filtered `adj` grid
    """
    rows, cols = [], []
    ref_by_cols, adj_by_cols = split_by_columns(ref), split_by_columns(adj)

    if len(ref_by_cols) != len(adj_by_cols):
        raise ValueError("Number of columns in 'ref' and "
                         "'adj' do not agree, {:d} != {:d}".format(
                             len(ref_by_cols), len(adj_by_cols)))

    for col_num in range(len(ref_by_cols[0])):
        for i in range(len(ref_by_cols[0][col_num])):
            ref_row = ref_by_cols[0][col_num][i]
            i_adj = find_index(adj_by_cols[0][col_num], ref_row)
            rows.append(adj_by_cols[0][col_num][i_adj])
            cols.append(adj_by_cols[1][col_num][i_adj])

    return (np.array(rows), np.array(cols))


def split_by_fringes(fancy_grid, row_total):
    """Split fancy grid by interference orders.

    This is different than `split_by_columns`, which only looks at column numbers.
    """
    N = fancy_grid[0].size//row_total
    rows, cols = fancy_grid
    new_rows = np.split(rows, N)
    new_cols = np.split(cols, N)

    return (new_rows, new_cols)


def split_by_columns(fancy_grid):
    """Split fancy grid into column-wise fancy-grid.

    Transforms a tuple of array (rows, cols) into a tuple of lists of rows
    (l_rows, l_cols), where l_rows (and l_cols) is a list arrays.
    """
    rows, cols = fancy_grid
    i_split = np.where(np.diff(rows) < 0)[0] + 1
    grid_by_columns = (np.split(rows, i_split), np.split(cols, i_split))

    return grid_by_columns


def teeth_per_stripe(fancy_grid):
    """Return the number of comb teeth per VIPA stripe as seen by the grid.

    Args:
    - rows: a sequence of row numbers with comb teeth sorted spectrally,
      i.e. the rows part of a fancy grid.

    Returns:
    - list containing the amount of comb teeth per stripe.
    """
    grid_by_columns = split_by_columns(fancy_grid)
    counts = np.array([col.size for col in grid_by_columns[0]])

    return counts


def make_grid_map(rows, cols):
    """Make a 2D bool array for visualizing VIPA diffraction pattern.
    """
    arr = np.zeros((dfcs_vipa.ROWS, dfcs_vipa.COLS), dtype=np.bool)
    arr[rows, cols] = True

    return arr


def linear_axis(grid_fancy, rio_pos, fsr, rio_lam, unit='wavenumber'):
    """Generate a frequency axis based on fringe-independent linear dispersion law.

    Parameters
    ----------
    grid_fancy : tuple of ndarray
        Two-element tuple, first element - rows of the grid, second
        element - columns of the grid.
    rio_pos : tuple of ndarray
        Tuple with two arrays containing the x, y coordinates of RIO dots.
    fsr : float
        Free spectral range of the VIPA etalon.
    rio_lam : float
        Wavelength in meters of the CW laser.
    unit : {'wavenumber', 'frequency', 'wavelength'}
        The unit of the frequency axis

    Returns
    -------
    ndarray
        The frequency axis of the spectrum.
    """
    rio_freq = lambda2nu(rio_lam)
    rio_index = closest_to_rio(rio_pos[1], grid_fancy)
    rio_order = np.round(rio_freq/fsr)
    row_total = rio_pos[0][1]-rio_pos[0][0]
    rio_column = rio_index // row_total
    column_total = grid_fancy[0]//row_total

    fringe_orders = np.arange(column_total) - rio_column + rio_order
    dispersion = fsr/row_total/rio_order
    
    

def naive_axis(grid_fancy, rio_pos, fsr, rio_lam, unit='wavenumber'):
    """Generate a frequency axis based on fringe-independent linear dispersion law.

    Parameters
    ----------
    grid_fancy : tuple of ndarray
        Two-element tuple, first element - rows of the grid, second
        element - columns of the grid.
    rio_pos : tuple of ndarray
        Tuple with two arrays containing the x, y coordinates of RIO dots.
    fsr : float
        Free spectral range of the VIPA etalon.
    rio_lam : float
        Wavelength in meters of the CW laser.
    unit : {'wavenumber', 'frequency', 'wavelength'}
        The unit of the frequency axis

    Returns
    -------
    ndarray
        The frequency axis of the spectrum.
    """
    rio_freq = lambda2nu(rio_lam)
    rio_index = closest_to_rio(rio_pos[1], grid_fancy)
    freq_axis = np.empty(grid_fancy[0].size)
    row_total = rio_pos[0][1]-rio_pos[0][0]
    dispersion = fsr/row_total

    freq_axis = (np.arange(grid_fancy[0].size)-rio_index)*dispersion+rio_freq

    if unit=='wavenumber':
        freq_axis = nu2wn(freq_axis)
    elif unit=='wavelength':
        freq_axis = nu2lambda(freq_axis)

    return freq_axis
    

if __name__ == '__main__':
    from pathlib import Path
    import h5py as h5
    try:
        grid_file = Path('/home/gkowzan/documents/nauka/fizyka/'
                         'DFCS/POMIARY/CCD/2017-05-19/lattice.hdf5')
        grid_file_dc = Path('/home/gkowzan/documents/nauka/fizyka'
                            '/DFCS/POMIARY/CCD/2017-05-19/lattice_dc.hdf5')
        grid_meas = h5.File(str(grid_file), mode='r')
        grid_meas_dc = h5.File(str(grid_file_dc), mode='r')

        grid_arrs = grid_meas['data'][...]-grid_meas_dc['data'][...]
        grid_arr = grid_arrs.mean(axis=0)
    finally:
        grid_meas.close()
        grid_meas_dc.close()

    grid = make_grid(grid_arr)
    grid_fancy = grid2fancy(grid)
    grid_map = make_grid_map(*grid_fancy)
