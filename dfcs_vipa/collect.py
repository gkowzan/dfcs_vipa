from pathlib import Path
import logging
import h5py as h5               # type: ignore
import numpy as np
import dfcs_vipa
from typing import TypeVar, Optional, Tuple, Sequence, List, Union

PathLike = TypeVar("PathLike", str, Path)

log = logging.getLogger(__name__)


# * Collect arrays and comb teeth intensities
def collect_h5(path: PathLike, path_dc: Optional[PathLike]=None) ->\
    np.ndarray:
    """Return all data from HDF5 DC measurements.

    Subtracts dark current from signal measurements.

    Parameters
    ----------
    path : PathLike
        Path to signal measurement file.
    path_dc: PathLike
        Path to dark current dark measurement file.

    Returns
    -------
    np.ndarray
        3D array with first dimension corresponds to different measurements and
    the last two corresponding to camera frame shape.
    """
    with h5.File(path, 'r') as f:
        if path_dc is not None:
            with h5.File(path_dc, 'r') as fdc:
                arrs = (f['data'][...].astype(np.int32)-fdc['data'][...].astype(np.int32))
        else:
            arrs = f['data'][...].astype(np.int32)

    return arrs


def average_h5(path: PathLike, path_dc: Optional[PathLike]=None) -> np.ndarray:
    """Return averaged data from HDF5 measurements.

    Subtracts dark current from the signal measurements.

    Parameters
    ----------
    path : PathLike
        Path to signal measurement file.
    path_dc: PathLike
        Path to dark current dark measurement file.

    Returns
    -------
    np.ndarray
        2D array shaped the same as camera frame.
    """
    with h5.File(path, 'r') as f:
        if path_dc is not None:
            with h5.File(path_dc, 'r') as fdc:
                arr = (f['data'][...].mean(axis=0) -
                       fdc['data'][...].mean(axis=0))
        else:
            arr = f['data'][...].mean(axis=0)

    return arr


def collect_element(path: PathLike,
                    row: np.integer, col: np.integer,
                    mask_cols: np.ndarray,
                    path_dc: Optional[PathLike]=None,
                    mask_rows: Optional[np.ndarray]=None)\
                    -> np.ndarray:
    """Collect single comb tooth intensities from data arrays.

    All pixels indexed by `mask_cols` and `mask_rows` relative to `row`, `col`
    are summed to botain comb tooth intensity.

    Parameters
    ----------
    path : PathLike
        Path to signal measurement file.
    row : int
        Row of comb tooth.
    col : int position of the comb tooth
        Column of comb tooth.
    mask_cols : ndarray
        Fancy indexing array for columns.
    path_dc: PathLike
        Path to dark current dark measurement file.
    mask_rows: ndarray
        Fancy indexing array for rows.

    Returns
    -------
    np.ndarray
        1D array containing comb tooth intensities.
    """
    # define the hyperslab
    col_min, col_max = col + mask_cols.min(), col + mask_cols.max() + 1
    if mask_rows is not None:
        row_min, row_max = row + mask_rows.min(), row + mask_rows.max() + 1
    else:
        row_min, row_max = row, row + 1

    # collect the hyperslab with the spectral element
    with h5.File(path, 'r') as f:
        if path_dc is not None:
            with h5.File(path_dc, 'r') as f_dc:
                element_array = (f['data'][..., row_min:row_max,
                                           col_min:col_max].astype(np.int32) -
                                 f_dc['data'][..., row_min:row_max,
                                              col_min:col_max].astype(np.int32))
        else:
            element_array = f['data'][..., row_min:row_max,
                                      col_min:col_max].astype(np.int32)

    # retrieve the data for a single spectral element
    if mask_rows is not None:
        rows = mask_rows - np.min(mask_rows)
    else:
        rows = np.zeros(mask_cols.shape, dtype=mask_cols.dtype)
    cols = mask_cols - np.min(mask_cols)

    elements = element_array[..., rows, cols].sum(axis=-1)

    return elements


def collect(arr: np.ndarray, grid_fancy: Tuple[np.ndarray, np.ndarray],
            mask_cols: np.ndarray, mask_rows: Optional[np.ndarray]=None)\
            -> np.ndarray:
    """Collect comb teeth intensities from data array.

    Parameters
    ----------
    arr: np.ndarray
        2D array of (averaged) camera frame.
    grid_fancy: tuple of ndarray
        Tuple of rows and cols array defining positions of the comb teeth.
    mask_cols: np.ndarray
        Fancy indexing array for columns.
    mask_rows: np.ndarray
        Fancy indexing array for rows.

    Returns
    -------
    np.ndarray
        1D array of comb teeth intensities.
    """
    rows, cols = grid_fancy
    cols = cols[:, np.newaxis] + mask_cols
    if mask_rows is not None:
        rows = rows[:, np.newaxis] + mask_rows
    else:
        rows = rows[:, np.newaxis]

    elements = arr[..., rows, cols]

    return elements.sum(axis=-1)


def collect_multi(ilist: Union[np.ndarray, Sequence[int]], fmt: str,
                  fmt_dc: Optional[str]=None) -> np.ndarray:
    """Collect averaged camera arrays from multiple files.

    Parameters
    ----------
    ilist : sequence
        Elements of `ilist` are formatted into `fmt` and `fmt_dc` strings to
        obtain paths to measurement files.
    fmt : str
        Format string for bright measurement.
    fmt_dc : str
        Format string for dark measurement.

    Returns
    -------
    np.ndarray
        3D array with first dimension corresponds to different measurements and
        the last two corresponding to camera frame dimensions.
    """
    ilength = len(ilist)
    multi_arr = np.empty((ilength, dfcs_vipa.ROWS, dfcs_vipa.COLS))
    for j, i in enumerate(ilist):
        log.info("Averaging '{:s}'".format(fmt.format(i)))
        bright_path = fmt.format(i)
        dark_path = None
        if fmt_dc is not None:
            dark_path = fmt_dc.format(i)
        multi_arr[j] = average_h5(bright_path, dark_path)

    return multi_arr


def collect_multi_single(ilist: Union[np.ndarray, Sequence[int]], fmt: str,
                  fmt_dc: Optional[str]=None) -> np.ndarray:
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
        bright_path = fmt.format(i)
        dark_path = None
        if fmt_dc is not None:
            dark_path = fmt_dc.format(i)
        multi_arr[j] = collect_h5(bright_path, dark_path)

    return multi_arr


def collect_multi_frep_scan(
        beat_range: np.ndarray,
        scan_range: Sequence[int],
        frep_range: Sequence[int],
        fmt: str, fmt_dc: Optional[str]=None,
        alt: Optional[bool]=True):
    """Collect averaged camera arrays from different freps and scans.

    The result is a nested dictionary with first level corresponding to
    different freps, the second level corresponding to different scans through
    the cavity modes.  Each (frep, scan) pair corresponds to a 3D NumPy array (a
    result of :func:`collect_multi`) with first dimension numbering different
    points on the cavity mode.

    Parameters
    ----------
    beat_range: ndarray of int
        an ndarray of ints numbering measurements within a cavity mode scan,
    scan_range: list of int
        list of ints numbering independent scans of a cavity mode,
    frep_range: list of ints
        a list of ints numbering different freps (jumps),
    fmt : str
    fmt_dc : str
        twice-nested format strings, the outer pattern corresponds to the frep
        number, the inner one to scan-beat number,
    alt: bool
        if True then the direction of cavity mode scan is reversed every second
        scan.

    Returns
    -------
    dict
        a twice-nested dictionary (frep=>scan=>collect_multi).
    """
    frep_arr_avgs = {}
    for l in frep_range:
        beat_arr_avgs = {}
        for k in scan_range:
            final_beat = (beat_range + k*40)
            if alt:
                if k % 2:
                    final_beat = final_beat[::-1]
            bright_path = fmt.format(l)
            dark_path = None
            if fmt_dc is not None:
                dark_path = fmt_dc.format(l)
            beat_arr_avgs[k] = collect_multi(
                final_beat,
                bright_path,
                dark_path)
        frep_arr_avgs[l] = beat_arr_avgs

    return frep_arr_avgs


def collect_multi_frep_scan_single(
        beat_range: np.ndarray,
        scan_range: Sequence[int],
        frep_range: Sequence[int],
        fmt: str, fmt_dc: Optional[str]=None,
        alt: Optional[bool]=True):
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
            bright_path = fmt.format(l)
            dark_path = None
            if fmt_dc is not None:
                dark_path = fmt_dc.format(l)
            beat_arr_avgs[k] = collect_multi_single(
                final_beat,
                bright_path,
                dark_path)
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
