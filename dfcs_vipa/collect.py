import logging
import h5py as h5
import numpy as np
import dfcs_vipa

log = logging.getLogger(__name__)


# * Collect arrays and comb teeth intensities
def collect_h5(path, path_dc):
    """Return all data from HDF5 DC measurements.

    Subtracts dark current from the signal measurements.

    Args:
    - path, path_dc: paths to signal and dark measurement files.

    Returns:
    - 3D array with first dimension corresponds to different measurements
      and the last two corresponding to camera array.
    """
    with h5.File(path, 'r') as f:
        with h5.File(path_dc, 'r') as fdc:
            arrs = (f['data'][...].astype(np.int32)-fdc['data'][...].astype(np.int32))

    return arrs


def average_h5(path, path_dc):
    """Return averaged data from HDF5 DC measurements.

    Subtracts dark current from the signal measurements.

    Args:
    - path, path_dc: paths to signal and dark measurement files.

    Returns:
    - 2D array containing averaged and DC-subtracted measurement.
    """
    with h5.File(path, 'r') as f:
        with h5.File(path_dc, 'r') as fdc:
            arr = (f['data'][...].mean(axis=0) -
                   fdc['data'][...].mean(axis=0))

    return arr


def collect_element(path, path_dc, row, col, mask_cols, mask_rows=None):
    """Collect single comb tooth intensities from data arrays.

    Args:
    - path, path_dc: paths to signal and dark measurement files,
    - row, col: position of the comb tooth,
    - mask_cols, mask_rows: numpy fancy indexing arrays defining single comb
      tooth pattern.

    Returns:
    - NumPy 1D array containing comb tooth intensities.
    """
    # define the hyperslab
    col_min, col_max = col + mask_cols.min(), col + mask_cols.max() + 1
    if mask_rows is not None:
        row_min, row_max = row + mask_rows.min(), row + mask_rows.max() + 1
    else:
        row_min, row_max = row, row + 1

    # collect the hyperslab with the spectral element
    with h5.File(path, 'r') as f:
        with h5.File(path_dc, 'r') as f_dc:
            element_array = (f['data'][..., row_min:row_max,
                                       col_min:col_max].astype(np.int32) -
                             f_dc['data'][..., row_min:row_max,
                                          col_min:col_max].astype(np.int32))

    # retrieve the data for a single spectral element
    if mask_rows is not None:
        rows = mask_rows - np.min(mask_rows)
    else:
        rows = np.zeros(mask_cols.shape, dtype=mask_cols.dtype)
    cols = mask_cols - np.min(mask_cols)

    elements = element_array[..., rows, cols].sum(axis=-1)

    return elements


def collect(arr, grid_fancy, mask_cols, mask_rows=None):
    """Collect comb teeth intensities from data array.

    Args:
    - arr: Numpy 2D array of (averaged) camera frame,
    - grid_fancy: tuple of rows and cols array defining positions of the
      comb teeth,
    - mask_cols, mask_rows: numpy fancy indexing arrays defining single comb
      tooth pattern.
    """
    rows, cols = grid_fancy
    cols = cols[:, np.newaxis] + mask_cols
    if mask_rows is not None:
        rows = rows[:, np.newaxis] + mask_rows
    else:
        rows = rows[:, np.newaxis]

    elements = arr[..., rows, cols]

    return elements.sum(axis=-1)


def collect_multi(ilist, fmt, fmt_dc):
    """Collect averaged camera arrays from multiple files.

    Args:
    - ilist: sequence which elements are formatted into fmt and fmt_dc to
      get paths to measurement files,
    - fmt, fmt_dc: format strings.

    Returns:
    - 3D array with first dimension corresponds to different measurements
      and the last two corresponding to camera frame dimensions.
    """
    ilength = len(ilist)
    multi_arr = np.empty((ilength, dfcs_vipa.ROWS, dfcs_vipa.COLS))
    for j, i in enumerate(ilist):
        log.info("Averaging '{:s}'".format(fmt.format(i)))
        multi_arr[j] = average_h5(fmt.format(i), fmt_dc.format(i))

    return multi_arr


def collect_multi_single(ilist, fmt, fmt_dc):
    """Collect camera arrays from multiple files.

    Parameters
    ----------
    ilist: sequence
        Sequence which elements are formatted into `fmt` and `fmt_dc` to
        get paths to measurement files.
    fmt : str
        Format string for bright frame.
    fmt_dc: str
        Format string for dark frame.

    Returns
    -------
    ndarray
        4D array with first dimension corresponding to different
        measurements, second dimension corresponding to different frames
        and the last two corresponding to camera frame dimensions.
    """
    ilength = len(ilist)
    with h5.File(fmt.format(ilist[0]), 'r') as test_file:
        frame_num = test_file['data'].shape[0]
    multi_arr = np.empty((ilength, frame_num, dfcs_vipa.ROWS, dfcs_vipa.COLS))
    for j, i in enumerate(ilist):
        log.info("Collecting '{:s}'".format(fmt.format(i)))
        multi_arr[j] = collect_h5(fmt.format(i), fmt_dc.format(i))

    return multi_arr


def collect_multi_frep_scan(beat_range, scan_range, frep_range, fmt, fmt_dc,
                            alt=True):
    """Collect averaged camera arrays from different freps and scans.

    The result is a nested dictionary with first level corresponding to
    different freps, the second level corresponding to different scans
    through the cavity modes.  Each frep, scan pair corresponds to a 3D
    NumPy array (a result of collect_multi) with first dimension numbering
    different points on the cavity mode.

    Args:
    - beat_range: a list of ints numbering measurements within a cavity mode
      scan,
    - scan_range: a list of ints numbering independent scans of a cavity
      mode,
    - frep_range: a list of ints numbering different freps (jumps),
    - fmt, fmt_dc: twice-nested format strings, the outer pattern
      corresponds to the frep number, the inner one to scan-beat number,
    - alt: if True then the direction of cavity mode scan is reversed every
      second scan.

    Returns:
    - twice-nested dictionary (frep=>scan=>collect_multi).
    """
    frep_arr_avgs = {}
    for l in frep_range:
        beat_arr_avgs = {}
        for k in scan_range:
            final_beat = (beat_range + k*40)
            if alt:
                if k % 2:
                    final_beat = final_beat[::-1]
            beat_arr_avgs[k] = collect_multi(
                final_beat,
                fmt.format(l),
                fmt_dc.format(l))
        frep_arr_avgs[l] = beat_arr_avgs

    return frep_arr_avgs


def collect_multi_frep_scan_single(beat_range, scan_range, frep_range, fmt, fmt_dc,
                                   alt=True):
    """Collect camera arrays from different freps and scans.

    The result is a nested dictionary with first level corresponding to
    different freps, the second level corresponding to different scans
    through the cavity modes and the third level corresponding to
    different beat note.  Each frep, scan pair corresponds to a 4D NumPy
    array (a result of collect_multi_single) with first dimension
    numbering different points on the cavity mode and the second one
    numbering different frames.

    Parameters
    ----------
    beat_range: list of int 
        A list of ints numbering measurements within a cavity mode scan,
    scan_range: list of int
        A list of ints numbering independent scans of a cavity mode,
    frep_range: list of int
        A list of ints numbering different freps (jumps),
    fmt : str
    fmt_dc : str 
        Twice-nested format strings, the outer pattern corresponds to
        the frep number, the inner one to scan-beat number,
    alt : bool 
        If True then the direction of cavity mode scan is reversed every
        second scan.

    Returns
    -------
    dict
        Twice-nested dictionary (frep=>scan=>beats), where `beats` is a
        4D array (beat=>frame=>row=>col).
    """
    frep_arr_avgs = {}
    for l in frep_range:
        beat_arr_avgs = {}
        for k in scan_range:
            final_beat = (beat_range + k*40)
            if alt:
                if k % 2:
                    final_beat = final_beat[::-1]
            beat_arr_avgs[k] = collect_multi_single(
                final_beat,
                fmt.format(l),
                fmt_dc.format(l))
        frep_arr_avgs[l] = beat_arr_avgs

    return frep_arr_avgs


def beat_range_list(beat_range, scan_range, alt=True):
    for k in scan_range:
        final_beat = (beat_range + k*40)
        if alt:
            if k % 2:
                final_beat = final_beat[::-1]
            for i in final_beat:
                yield i


# * Noise analysis
def frame_stdevs_h5(path):
    """Return standard deviations of each frame.
    """
    with h5.File(path, 'r') as f:
        stdevs = frame_stdevs(f['data'][...])

    return stdevs


def threshold_stdevs(stdevs, scale):
    return stdevs.mean() + scale*np.std(stdevs)


def frame_stdevs(arr):
    mean = arr.mean(axis=0)

    return np.sum(np.sum((arr-mean)**2, axis=-1), axis=-1)


def spectra_stdevs(arr, grid_fancy, mask_cols, mask_rows=None):
    spectra = collect(arr, grid_fancy, mask_cols, mask_rows)
    spectra_mean = spectra.mean(axis=0)

    return np.sum((spectra-spectra_mean)**2, axis=-1)


# * Collect frequency counter measurements
counter_dtype = np.dtype({
    'names':
    ['timestamp_abs', 'reprate_freq', 'offset_freq', 'counter2_freq'],
    'formats': ['f8']*4
})


def collect_counter(path):
    return np.genfromtxt(path, dtype=counter_dtype, skip_header=1,
                         delimiter=',')


def collect_counters(fmt, indices):
    return np.hstack([collect_counter(fmt.format(i)) for i in indices])
