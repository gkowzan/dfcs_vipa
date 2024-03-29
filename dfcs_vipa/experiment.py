"""Functions helpful in analyzing experimental data."""
from warnings import warn
import numpy as np
from scipy.signal import argrelextrema


# * Miscellaneous
def remove_close(maxima, distance, *arrays):
    """Remove points in `maxima` which are closer than `distance`.

    It is assumed that the first maximum is a proper one.

    Parameters
    ----------
    maxima : ndarray
    distance : float
    arrays : list of ndarray, optional
        List of other arrays from which elements at the same positions
        as the unwanted elements in `maxima` shoudl also be removed.

    Returns
    -------
    maxima : ndarray
    arrays : list of ndarray, optional
    """
    indices = []
    i = 0
    while i < len(maxima) - 1:
        if maxima[i+1] - maxima[i] < distance:
            indices.append(i+1)
            i += 1
        i += 1
    maxima = np.delete(maxima, indices)
    if arrays:
        arrays = [np.delete(arr, indices) for arr in arrays]
    if len(indices) == 0:
        if arrays:
            return maxima, arrays
        else:
            return maxima
    else:
        return remove_close(maxima, distance, *arrays)


def expand(arr, n=1):
    """Add `n` evenly spaced points to the start and end of `arr`.

    Assumes points in `arr` are evenly spaced.  Original array is
    retrieved by ret[n:-n] slice.

    Parameters
    ----------
    arr : ndarray
    n : int, optional

    Returns
    -------
    ret : ndarray
    """
    dx = arr[1]-arr[0]
    add = np.arange(1, n+1)*dx

    return np.concatenate((-add[::-1]+arr[0], arr, arr[-1]+add))


def denser(arr, n):
    """Add `n` points between each point in `arr`.

    Assumes points in `arr` are evenly spaced.  Original array is
    retrieved by ret[::n+1] slice.

    Parameters
    ----------
    arr : ndarray
    n : int

    Returns
    -------
    ret : ndarray

    """
    span = arr[1] - arr[0]
    mid = np.linspace(0, span, n+2)[:-1]
    ret = arr[:, np.newaxis] + mid[np.newaxis, :]

    return ret.flatten()[:-n]


def find_maxima(y, window_len=10, thres=0, order=3):
    """Find maxima in 1D array 'y'.

    Smooth the data with hanning window of length 'window_len' and threshold
    it.
    """
    y = y - thres
    y[y < 0] = 0
    ysmooth = smooth(y, window_len=window_len)

    return argrelextrema(ysmooth, np.greater, order=order)[0]


def find_maxima_abs(y, window_len=10, thres=0, order=3):
    y = smooth(np.abs(y), window_len)
    if thres=='auto':
        thres = y.max()/2
    y[y < thres] = 0
    
    return argrelextrema(y, np.greater, order=order)[0]


def find_minima(y, window_len=10, thres=0.0, order=3):
    """Find maxima in 1D array 'y'.

    Smooth the data with hanning window of length 'window_len' and threshold
    it.
    """
    y = y - thres
    y[y < 0] = 0
    ysmooth = smooth(y, window_len=window_len)

    return argrelextrema(ysmooth, np.less, order=order)[0]


def same_sign(arr):
    """Check if all elements have the same sign."""
    return np.all(arr > 0) if arr[0] > 0 else np.all(arr < 0)


def find_indices(arr, vals):
    """Return indices of `arr` closest to `vals`."""
    return np.argmin(np.abs(arr[np.newaxis, :] - vals[:, np.newaxis]), axis=1)


def find_index(arr, val, axis=None):
    """Returns index of an `arr` value that is closest to `val`."""
    return np.argmin(np.abs(arr-val), axis=axis)


def normalized(y):
    '''Return array normalized to max value.'''
    return y/np.max(y)


def amp(y):
    return np.max(y)-np.min(y)


def running_mean(x, n=2):
    """Running average of 'n' samples.

    Returns array of size max(x.size, n) - min(x.size, n) + 1"""
    window = np.ones(n, 'd')

    return np.convolve(window/window.sum(), x, mode='valid')


def smooth(x, window_len=10, window='hanning'):
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x
        Input signal
    window_len : optional
        Dimension of the smoothing window
    window : optional
        The type of the window: 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman'.

    Returns
    -------
    np.ndarray
        Smoothed signal.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', "
                         "'bartlett', 'blackman'")

    s = np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    # print(len(s))

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


def fwhm_est(x, y, background=True):
    """Estimate FWHM from x, y data."""
    if np.iscomplexobj(y):
        warn("Input array of 'fwhm_est' is complex, the function may not work as expected!")
    sorter = np.argsort(x)
    x = x[sorter]
    y = y[sorter]

    i_max = np.argmax(y)
    if background:
        halfmax = (np.max(y)-np.min(y))/2 + np.min(y)
    else:
        halfmax = np.max(y)/2
    i_half_left = find_index(y[:i_max], halfmax)
    i_half_right = find_index(y[i_max:], halfmax) + i_max

    return (x[i_half_right]-x[i_half_left]), i_half_left, i_half_right, i_max


def linspace_with(start, stop, num=50, endpoint=True, dtype=None,
                  include=None):
    ret = np.linspace(start, stop, num, endpoint, dtype=dtype)
    include = np.asarray(include)
    ret = np.union1d(ret, include)

    return ret


def arange_with(*args, dtype=None, include=None):
    ret = np.arange(*args, dtype=dtype)
    include = np.asarray(include)
    ret = np.union1d(ret, include)

    return ret


def struct2dict(struct, d=None, exclude=tuple()):
    """Update 'd' with values from structured array 'struct'.

    Don't copy fields in 'exclude'.

    Returns:
    - updated 'd' dictionary"""
    if d is None:
        d = {}
    for field, type in struct.dtype.fields.items():
        if field not in exclude:
            if type[0] is np.bytes_:
                d[field] = struct[field][0].decode().strip()
            else:
                d[field] = struct[field][0]

    return d


# * Allan deviation
def _square_difference(j, arr, tau):
    return (arr[(j+1)*tau:(j+2)*tau].mean()-arr[j*tau:(j+1)*tau].mean())**2


def _allan_tau(arr, tau):
    n = arr.size//tau-1
    r = 0.0

    for j in range(n):
        r += _square_difference(j, arr, tau)

    return r/2/n


def allan_points(N):
    """Allan deviation averaging spans for N-sized sample.

    Allan deviation for the longest averaging span is calculated based
    on at least 5 spans.

    Parameters
    ----------
    N : int
        Number of samples.

    Returns
    --------
    ns : ndarray
        Averaging spans.

    """
    max_n = N//5
    cur_p = 0
    max_p = np.int(np.log10(max_n))
    ns = []
    while cur_p <= max_p:
        for i in range(1, 10):
            n = i*10**cur_p
            if n <= max_n:
                ns.append(n)
        cur_p += 1

    return np.array(ns)


def allan_variance(arr):
    """Returns Allan variance of samples in `arr`.

    Values in array are taken as corresponding to consecutive and evenly
    spaced time periods.  Variance is calculated for averaging spans between 1 and
    arr.size//5-long, with 10 points per order of magnitude.
    """
    points = allan_points(arr.size)
    r = np.empty(points.size, dtype=np.double)
    for tau, i in zip(points, range(points.size)):
        r[i] = _allan_tau(arr, tau)

    return points, r


def allan_deviation(arr):
    points, avar = allan_variance(arr)

    return points, np.sqrt(avar)


# * LabView arrays
def read1dlv(filename):
    """Read LabView 1D array and return as numpy array.

    Little-endian version.
    """
    if isinstance(filename, str):
        f = open(filename, 'rb')
    else:
        f = filename
    dim = np.fromstring(f.read(4), dtype=np.dtype('u4'))
    arr = np.fromstring(f.read(), count=dim[0], dtype=np.float)

    return arr


def read2dlv(filename):
    """Read LabView 2D array and return as numpy array.

    Big-endian version.
    """
    if isinstance(filename, str):
        f = open(filename, 'rb')
    else:
        f = filename
    dims = np.fromstring(f.read(8), dtype=np.dtype('>u4'))
    ccd_temp = np.fromstring(f.read(), dtype=np.dtype('>u2'))
    ccd_arr = ccd_temp.astype(np.uint16).reshape(dims)
    f.close()

    return ccd_arr


def write2dlv(arr, filename):
    """Write numpy array to file in LabView format."""
    if isinstance(filename, str):
        with open(filename, 'w'):
            pass
        f = open(filename, 'a')
    else:
        f = filename
    np.array(arr.shape, dtype=np.dtype('>u4')).tofile(f)
    arr.astype('>u2', copy=False).tofile(f)
    f.close()
