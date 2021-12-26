"""Calibrate (InGaAs) camera nonlinearity.

Uses a series of measurements of a Gaussian beam/flat field imaged on the
camera with different integration times to establish the real dependence
between the energy incident upon the camera and the number of counts
returned by the acquisition software.

The camera is linear only in a very small range at very low collected energies.
The calibration function is used in following way::

    calibrated = [cam*cal_func(cam) for cam in cameras]

where cameras is a list of camera images for different integration times.
"""
import codecs
import logging
from itertools import product
from typing import Sequence, Callable, Tuple

import numpy as np
from lxml import etree
from scipy.interpolate import splev, splrep
from scipy.optimize import curve_fit

import dfcs_vipa
from dfcs_vipa.collect import average_h5

log = logging.getLogger(__name__)
default_nodes = np.hstack([
    np.linspace(0, 150, 15),
    np.linspace(150, 4100, 20)])


def linear(x, a):
    return a*x


def read_time(fmt_string_xcf, num):
    log.debug('reading integration time {:d}'.format(num))
    path = fmt_string_xcf.format(num)
    with codecs.open(str(path), 'r', 'utf-8') as f:
        xml_string = '\n'.join(f.readlines()[1:])
    tree = etree.fromstring(xml_string)

    return float(tree[0].get('value'))


def read_cam_avg(fmt_string, fmt_string_dc, num):
    log.debug('reading camera average {:d}'.format(num))
    path = fmt_string.format(num)
    path_dc = fmt_string_dc.format(num)

    return average_h5(path, path_dc)


class ReadDataMixin:
    def __init__(self, fmt_string: str, fmt_string_dc: str,
                 fmt_list: Sequence[int]):
        """Helper functions to read camera frame data.

        `format` method will be applied to `fmt_string` and `fmt_string_dc` for
        each element of `fmt_list` to obtain actual bright frames, dark frames
        and XML metadata files.

        Parameters
        ----------
        fmt_string
            File name of bright camera frame without extension.
        fmt_string_dc
            File name of dark camera frame without extension.
        fmt_list
            Sequence of integers numbering calibration measurements.
        """
        self.fmt_string = fmt_string + '.hdf5'
        self.fmt_string_dc = fmt_string_dc + '.hdf5'
        self.fmt_string_xcf = fmt_string_dc + '.xcf'
        self.fmt_list = fmt_list
        self.times = None
        self.cameras = None

    def read_time(self, num):
        """Read integration time from XML metadata file."""
        # log.debug('reading integration time {:d}'.format(num))
        return read_time(self.fmt_string_xcf, num)

    def read_times(self):
        """Read all integration times."""
        log.info('reading integration times')
        self.times = np.array(
            [self.read_time(i) for i in self.fmt_list]
        )

    def read_cam_avg(self, num):
        """Load and average bright and dark frames for `num`."""
        # log.debug('reading camerage average {:d}'.format(num))
        return read_cam_avg(self.fmt_string, self.fmt_string_dc, num)

    def read_cameras(self):
        """Load and average all bright and dark frames."""
        log.info('reading camera averages')
        self.cameras = [self.read_cam_avg(i) for i in self.fmt_list]


class PixelCalibrate(ReadDataMixin):
    def __init__(self, fmt_string: str, fmt_string_dc: str,
                 fmt_list: Sequence[int]):
        """Calibrate linearity of each camera pixel individually.

        Arguments are the same as for :class:`ReadDataMixin`.
        """
        super(PixelCalibrate, self).__init__(fmt_string, fmt_string_dc,
                                             fmt_list)
        self.cameras = None
        self.calibration = None
        self.mask = set()

    def calibrate(self, init_range: slice):
        """Read in all metadata and camera frames, prepare calibration matrix."""
        self.read_times()
        self.read_cameras()
        self.fit(init_range)

    def read_cameras(self):
        """Load and average all bright and dark frames."""
        log.info('reading camera averages')
        self.cameras = np.empty((len(self.fmt_list),
                                 dfcs_vipa.ROWS, dfcs_vipa.COLS))
        for i, num in enumerate(self.fmt_list, start=0):
            self.cameras[i] = self.read_cam_avg(num)

    def fit_pixel(self, x: np.ndarray, y: np.ndarray, init_range: slice) ->\
        Tuple[np.ndarray, np.ndarray]:
        """Fit `int_time->count_rate` of a pixel to linear dependence.

        The fitting range is limited to `init_range`. The residual between the
        linear dependence and actual dependence (fitted to a cubic spline) gives
        the calibration curve.

        Parameters
        ----------
        x
            Integration times.
        y
            Counts.
        init_range
            Indices into `x` and `y`.

        Returns
        -------
        t, c
            Knots and coefficients of cubic spline, as returned by
            :func:`splrep`.
        """
        xi, yi = x[init_range], y[init_range]
        a = curve_fit(linear, xi, yi)[0][0]
        res = linear(x, a)/y

        return splrep(y, res, k=1, task=0, s=0.0)[:2]

    @staticmethod
    def monotonize(t: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        idx = np.unique(np.maximum.accumulate(p), return_index=True)[1]

        return t[idx], p[idx]

    def fit(self, init_range: slice):
        """Calculate calibration curves for all pixels.

        See :meth:`fit_pixel`.
        """
        log.info(('fitting pixel nonlinearity, '
                  'initial range: {!r}').format(init_range))
        self.calibration = dict()
        for i, j in product(range(dfcs_vipa.ROWS), range(dfcs_vipa.COLS)):
            if (i, j) in self.mask:
                log.info('skipping pixel ({:d}, {:d})'.format(i, j))
            else:
                log.debug('fitting pixel ({:d}, {:d})'.format(i, j))

                t, p = self.monotonize(self.times, self.cameras[:, i, j])
                self.calibration[(i, j)] = self.fit_pixel(
                    t, p, init_range
                )

    def cal_matrix(self, cam: np.ndarray) -> np.ndarray:
        """Return the multiplicative calibration matrix for `cam`."""
        log.info('calculating calibration matrix')
        ret = np.full(cam.shape, 1.0)
        for p, (knots, coeffs) in self.calibration.items():
            ret[p] = splev(cam[p], (knots, coeffs, 1), ext=3)

        return ret

    def single_pixel(self, row, col) -> Callable[[np.ndarray], np.ndarray]:
        """Return a calibrating function based on a single-pixel fit."""
        knots, coeffs = self.calibration[(row, col)]

        def cal_func(cam: np.ndarray):
            return splev(cam, (knots, coeffs, 1), ext=3)

        return cal_func

    def __call__(self, cam: np.ndarray) -> np.ndarray:
        return self.cal_matrix(cam)
