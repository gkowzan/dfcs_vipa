"""Calibrate InGaAs camera nonlinearity.

Uses a series of measurements of a Gaussian beam/flat field imaged on the
camera with different integration times to establish the real dependence
between the energy incident upon the camera and the number of counts
returned by the acquisition software.

The camera is linear only in a very small range at very low collected energies.
The calibration function is used in following way:
calibrated = [cam*cal_func(cam) for cam in cameras]
where cameras is a list of camera images for different integration times.
"""
import logging
from itertools import combinations, product
from collections import namedtuple
import codecs
from lxml import etree
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import (interp1d, splrep, splev)
from dfcs_vipa.collect import average_h5
from dfcs_vipa.experiment import smooth, find_index
import dfcs_vipa

log = logging.getLogger(__name__)
default_nodes = np.hstack([
    np.linspace(0, 150, 15),
    np.linspace(150, 4100, 20)])


def linear(x, a):
    return a*x


def rational(x, a):
    return 1/(1 + a*x)


###############################################################################
# piecewise linear function for RatioPiecewise class calibration              #
###############################################################################
def piecewise_linear(x, nodes, slopes, fill_value=1.0):
    """Evaluate piecewise linear function.

    The function is defined for values between min(nodes) and
    max(nodes).  The intercept of the first segment is fixed to one and
    the intercepts of consecutive segments are fixed to max values of
    the previous segments.

    The intervals of each linear segment are half-closed [nodes[i],
    nodes[i+1]).  The segments are evaluated by calculating

    slopes[i]*(x-nodes[i])+intercepts[i]

    for corresponding segments.  That is, the 'x' values are shifted
    before applying the slope.

    Parameters
    ----------
    x : float or array_like
        The points for which the function should be evaluated.
    nodes : ndarray
        Positions of the nodes separating segments.
    slopes : ndarray
        Linear slopes of segments.
    fill_value : float
        Value for points beyond [min(nodes), max(nodes)].

    Returns
    -------
    float or array_like
        Values of piecewise linear function at `x`.
    """
    if len(slopes) != len(nodes) - 1:
        raise ValueError("Length of 'slopes' should be one less than length of"
                         " 'nodes'")

    # calculate the intercepts
    intercepts = np.ones(len(slopes))
    for i in range(1, len(slopes)):
        intercepts[i] = slopes[i-1]*(nodes[i]-nodes[i-1]) + intercepts[i-1]

    # evaluate the function
    result = np.full(len(x), fill_value)
    # idx = np.empty(len(x), dtype=np.bool)
    for i in range(len(slopes)):
        # print(x)
        # print(nodes[i])
        # idx[:] = np.logical_and(x >= nodes[i], x < nodes[i+1])
        idx = np.where(np.logical_and(x >= nodes[i], x < nodes[i+1]))
        result[idx] = slopes[i]*(x[idx]-nodes[i]) + intercepts[i]

    return result


###############################################################################
# helper functions for reading measurement data                               #
###############################################################################
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


###############################################################################
# Mix-in class for reading measurement data                                   #
###############################################################################
class ReadDataMixin:
    def __init__(self, fmt_string, fmt_string_dc, fmt_list):
        self.fmt_string = fmt_string + '.hdf5'
        self.fmt_string_dc = fmt_string_dc + '.hdf5'
        self.fmt_string_xcf = fmt_string_dc + '.xcf'
        self.fmt_list = fmt_list
        self.times = None
        self.cameras = None

    def read_time(self, num):
        # log.debug('reading integration time {:d}'.format(num))
        return read_time(self.fmt_string_xcf, num)

    def read_times(self):
        log.info('reading integration times')
        self.times = np.array(
            [self.read_time(i) for i in self.fmt_list]
        )

    def read_cam_avg(self, num):
        # log.debug('reading camerage average {:d}'.format(num))
        return read_cam_avg(self.fmt_string, self.fmt_string_dc, num)

    def read_cameras(self):
        log.info('reading camera averages')
        self.cameras = [self.read_cam_avg(i) for i in self.fmt_list]


###############################################################################
# class for minimizing sigma of Gaussian beam ratios                          #
###############################################################################
FitResult = namedtuple('FitResult', 'x cov_x infodict mesg ier')


class RatioCalibration(ReadDataMixin):
    def __init__(self, fmt_string, fmt_string_dc, fmt_list,
                 initial_coeffs=None):
        super(RatioCalibration, self).__init__(fmt_string, fmt_string_dc,
                                               fmt_list)
        self.pairs = list(combinations(fmt_list, 2))
        self.coeffs = None
        if initial_coeffs is None:
            self.skip_initial = False
            self.initial_coeffs = np.full(len(self.nodes)-1, 1e-5)
        else:
            self.skip_initial = True
            self.initial_coeffs = initial_coeffs
        self._window = (slice(None, dfcs_vipa.ROWS), slice(None, dfcs_vipa.COLS))
        self.total = dfcs_vipa.ROWS*dfcs_vipa.COLS
        self.res = None

    def corr_func(self, coeffs, x):
        raise NotImplementedError("Implement this method in a subclass.")

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, val):
        self._window = val
        rows = val[0].stop-val[0].start
        cols = val[1].stop-val[1].start
        self.total = rows*cols

    def initial_fit(self, pos, fit_range=slice(None, None)):
        log.info('performing the initial ratio fit')
        powers = np.array(
            [cam[pos] for cam in self.cameras]
        )
        popt, pcov = curve_fit(linear, self.times[fit_range],
                               powers[fit_range])
        res = linear(self.times, *popt)/powers

        def res_func(coeffs):
            return self.corr_func(coeffs, powers)-res

        coeffs = least_squares(
            res_func,
            self.initial_coeffs,
            bounds=(0.0, 2*0.00075))
        # coeffs = leastsq(res_func, self.initial_coeffs, full_output=True)[0]
        self.initial_coeffs = coeffs.x

    def residuals_old(self, coeffs):
        # apply the correction
        rows = self.window[0]
        cols = self.window[1]
        fixed = [cam[rows, cols]*self.corr_func(coeffs, cam[rows, cols])
                 for cam in self.cameras]

        # calculate the residuals
        len_pairs = len(self.pairs)
        if self.res is None:
            self.res = np.empty(self.total*len_pairs)

        for i, (ip1, ip2) in zip(range(len_pairs), self.pairs):
            ilow, ihigh = i*self.total, (i+1)*self.total
            self.res[ilow:ihigh] = (fixed[ip1]/fixed[ip2]).reshape(-1)
            self.res[ilow:ihigh] = self.res[ilow:ihigh]-self.res[ilow:ihigh].mean()

        return self.res

    def narrow_cameras(self):
        rows = self.window[0]
        cols = self.window[1]
        self.cameras_narrowed = np.empty((len(self.fmt_list), self.total))
        for i in range(len(self.cameras)):
            self.cameras_narrowed[i] = self.cameras[i][rows, cols].reshape(-1)

    def correct(self, coeffs):
        # apply the correction
        cams = self.cameras_narrowed
        return cams*self.corr_func(coeffs, cams)

    def residuals(self, coeffs):
        # apply the correction
        fixed = self.correct(coeffs)

        # calculate the residuals
        len_pairs = len(self.pairs)
        # len_pairs = len(self.fmt_list)-1
        if self.res is None:
            self.res = np.empty(self.total*len_pairs)

        # print((fixed[-1]/fixed[0]).reshape(-1))
        for i, (ip1, ip2) in zip(range(len_pairs), self.pairs):
        # for i in range(len_pairs):
            ilow, ihigh = i*self.total, (i+1)*self.total
            # self.res[ilow:ihigh] = (fixed[-1]/fixed[i]).reshape(-1)
            self.res[ilow:ihigh] = (fixed[ip1]/fixed[ip2]).reshape(-1)
            self.res[ilow:ihigh] = self.res[ilow:ihigh]/self.res[ilow:ihigh].mean()-1
        # print(res)
        return np.copy(self.res)

    def fit(self, **kwargs):
        log.info('performing the main ratio fit')
        self.fit_results = least_squares(
            self.residuals,
            self.initial_coeffs,
            bounds=(0.0, 2*0.00075),
            verbose=2,
            max_nfev=10,
            # epsfcn=1e-2,
            **kwargs)

        def cal_func(cam):
            return self.corr_func(self.fit_results.x, cam)

        return cal_func

    def calibrate(self, pos, init_range, **kwargs):
        self.read_times()
        self.read_cameras()
        self.narrow_cameras()
        if not self.skip_initial:
            self.initial_fit(pos, init_range)
        self.cal_func = self.fit(**kwargs)


class RatioCalibrationPiecewise(RatioCalibration):
    def __init__(self, *args, **kwargs):
        self.nodes = np.hstack([
            np.linspace(-10, 150, 40),
            np.arange(150, 360, 15),
            np.linspace(360, 3000, 20),
            np.linspace(3000, 4100, 25)
        ])
        super(RatioCalibrationPiecewise, self).__init__(*args, **kwargs)

    def corr_func(self, coeffs, x):
        c = x.view()
        c.shape = (np.prod(x.shape))
        ret = piecewise_linear(c, self.nodes, coeffs)

        return ret.reshape(x.shape)


#############################################################################
# functions for rescaling different measurements and fitting nonlinearity   #
# directly                                                                  #
#############################################################################
class RescaleTimesMinimum:
    def __init__(self, ref_times, ref, other_times, other):
        self.ref_times = ref_times
        self.ref = ref
        self.other_times = other_times
        self.other = other
        self.ref_func = interp1d(ref_times, ref, bounds_error=True)
        self.t_max = ref_times.max()
        self.s_max = ref_times.max()/other_times.min()  # max. scale factor
        self.s_min = ref_times.min()/other_times.max()
        self.fit_results = None

    def residuals(self, scale, res_points):
        """Calculate residuals for least squares fitting."""
        t_scaled = self.other_times*scale
        t_min = max([min(t_scaled), min(self.ref_times)])
        t_max = min([max(t_scaled), max(self.ref_times)])
        t_res = np.linspace(t_min, t_max, res_points)
        other_func = interp1d(t_scaled, self.other, bounds_error=True)

        return self.ref_func(t_res)-other_func(t_res)

    def rescale_time(self, res_points=1000, guess=20):
        """Match 'other' curve to 'ref' by rescaling int. times.

        Args:
        - res_points: number of residuals.

        Return:
        - rescaled 'times'"""
        self.fit_results = least_squares(
            self.residuals, guess,
            bounds=(self.s_min, self.s_max),
            args=(res_points, ),
            verbose=0,
            jac='3-point',
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
            loss='cauchy'
        )
        if self.fit_results.cost > 1000:
            raise RuntimeError('Fit did not converge, cost = %{:.2f}'.format(
                self.fit_results.cost
            ))
        # plt.figure()
        # plt.plot(self.ref_times, self.ref)
        # plt.plot(self.other_times*self.fit_results.x[0], self.other)
        # t_min = max([min(self.other_times*self.fit_results.x[0]),
        #              min(self.ref_times)])
        # t_max = min([max(self.other_times*self.fit_results.x[0]),
        #              max(self.ref_times)])
        # plt.axvline(x=t_min)
        # plt.axvline(x=t_max)
        # t_res = np.linspace(t_min, t_max, res_points)
        # plt.plot(t_res, self.ref_func(t_res), 'ro')

        return self.other_times*self.fit_results.x[0]


class RescaleTimes:
    def __init__(self, ref_times, ref, other_times, other):
        self.ref_times = ref_times
        self.ref = ref
        self.other_times
        self.other = other
        self.ref_func = interp1d(ref_times, ref, bounds_error=True)
        self.t_min = np.min(ref_times)
        self.s_max = np.max(ref_times)/self.t_min  # max. scale factor
        self.fit_results = None

    def residuals(self, scale, res_points):
        """Calculate residuals for least squares fitting."""
        t_scaled = self.other_times/scale
        t_max = np.max(t_scaled)
        t_res = np.linspace(self.t_min, t_max, res_points)
        other_func = interp1d(t_scaled, self.other, bounds_error=True)

        return self.ref_func(t_res)-other_func(t_res)

    def rescale_time(self, res_points=1000, guess=20):
        """Match 'other' curve to 'ref' by rescaling int. times.

        Args:
        - res_points: number of residuals.

        Return:
        - rescaled 'times'"""
        self.fit_results = least_squares(
            self.residuals, guess,
            bounds=(1.0, self.s_max),
            args=(res_points, ),
            verbose=0,
            jac='3-point',
            loss='soft_l1',
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12
        )

        # plt.figure()
        # plt.plot(self.times, self.ref)
        # plt.plot(self.times/self.fit_results.x[0], self.other)

        return self.other_times/self.fit_results.x[0]


###############################################################################
# class for calibrating nonlinearity directly from time-power dependence      #
###############################################################################
class DirectCalibration(ReadDataMixin):
    """Provides rescaling function to linearize raw data.

    Uses the dependence of digital counts on integration time at constant
    illumination to calculate the deviation from linearity.  Rescales the time
    dependence from different pixels to obtain more data points.
    """
    def __init__(self, fmt_string, fmt_string_dc, fmt_list,
                 nodes=default_nodes):
        super(DirectCalibration, self).__init__(fmt_string, fmt_string_dc,
                                                fmt_list)
        self.gathered = None
        self.nodes = nodes
        self.initial_coeffs = np.full(len(self.nodes)-1, 1e-5)
        self.cal_func = None

    def gather(self, pos_mins):
        # powers_max = [cam[pos_max] for cam in self.cameras]
        powers_min = [[cam[pos_min] for cam in self.cameras]
                      for pos_min in pos_mins]
        powers_sample = np.array([cameras[-1] for cameras in powers_min])
        powers_max = powers_min.pop(np.argmin(powers_sample))

        times = self.times
        powers = powers_max
        while powers_min:
            power_min = powers_min.pop()
            try:
                times_min = RescaleTimesMinimum(
                    times,
                    powers,
                    self.times,
                    power_min
                ).rescale_time()
            except RuntimeError:
                log.info('skipping pixel')
                continue
            times = np.hstack([times_min, times])
            powers = np.hstack([power_min, powers])
            sorter = np.argsort(times)
            times = times[sorter]
            powers = powers[sorter]

        self.gathered = (times, powers)

    def calibrate(self, pos_min, init_range, *args, **kwargs):
        self.read_times()
        self.read_cameras()
        self.gather(pos_min)
        self.fit(init_range, *args, **kwargs)

    def fit(self, init_range, *args, **kwargs):
        times = self.gathered[0]
        powers = self.gathered[1]
        x, y = times[init_range], powers[init_range]
        a = curve_fit(linear, x, y)[0][0]
        # popt = curve_fit(rational, x, y, p0=(1, 1, 1, 1000))[0]

        # plt.figure()
        # plt.plot(times, powers, label='data')
        # plt.plot(times, linear(times, a), label='initial fit')
        # plt.plot(times, rational(times, *popt), label='initial fit')

        res = linear(times, a)/powers
        # res = powers/linear(times, a)
        # res = rational(times, *popt)/powers
        res_smooth = np.empty(res.size)
        i_1000 = find_index(powers, 3200)
        res_smooth[:i_1000] = smooth(res[:i_1000], window_len=50)
        res_smooth[i_1000:] = res[i_1000:]
        # i_100, i_500 = find_index(powers, 50), find_index(powers, 150)
        # res_smooth[:i_500] = res_smooth[:i_500]*0.95
        res_smooth = res_smooth

        # plt.figure()
        # plt.plot(powers, res)
        # plt.plot(powers, res_smooth)

        # plt.figure()
        # plt.plot(powers, res-res_smooth)

        self.cal_func = interp1d(powers, res_smooth,
                                 assume_sorted=False,
                                 bounds_error=False,
                                 fill_value=(res_smooth.min(),
                                             res_smooth.max()))

        # popt = curve_fit(rational, powers, res_smooth,
        #                  p0=(0.01, ))[0]
        # print(popt)
        # plt.figure()
        # plt.plot(powers, rational(powers, *popt))
        # plt.plot(powers, res_smooth)
        # self.cal_func = LSQUnivariateSpline(
        #     powers, res_smooth, nodes,
        #     ext=0,          # extrapolate
        #     *args, **kwargs
        # )

    def __call__(self, x):
        return self.cal_func(x)

    # def corr_func(self, x, coeffs):
    #     c = x.view()
    #     c.shape = (np.prod(x.shape))
    #     ret = piecewise_linear(c, self.nodes, coeffs)

    #     return ret.reshape(x.shape)

    # def residuals(self, coeffs, powers, res):
    #     return self.corr_func(powers, coeffs)-res

    # def corr_func(self, x, spline, low, high):
    #     x_min = self.gathered[0].min()
    #     x_max = self.gathered[1].max()

    #     y_min = spline(x_min)
    #     a_min = y_min/x_min

    #     y_max = spline(x_max)


def column_calibrations(rows, cols, fmt_string, fmt_string_dc, fmt_list):
    corrections = []
    for col in cols:
        log.info('calibrating column {:d}'.format(col))
        calibration = DirectCalibration(fmt_string, fmt_string_dc, fmt_list)
        calibration.calibrate(
            [(row, col) for row in rows],
            slice(None, 10)
        )
        calibration.cameras = []
        corrections.append(
            ((slice(None, None), col),
             calibration)
        )

    return corrections


class CalibrationMatrix:
    def __init__(self, corrections):
        self.corrections = corrections

    def __call__(self, array):
        """Return same-size array to be mulitplied with 'array'."""
        ret = np.full(array.shape, 1.0)
        for slices, func in self.corrections:
            ret[slices[0], slices[1]] = func(array[slices[0], slices[1]])

        return ret


###############################################################################
# calibrate each pixel separately                                             #
###############################################################################
class PixelCalibrate(ReadDataMixin):
    def __init__(self, fmt_string, fmt_string_dc, fmt_list):
        super(PixelCalibrate, self).__init__(fmt_string, fmt_string_dc,
                                             fmt_list)
        self.cameras = None
        self.calibration = None
        self.mask = set()

    def calibrate(self, init_range):
        self.read_times()
        self.read_cameras()
        self.fit(init_range)

    def read_cameras(self):
        log.info('reading camera averages')
        self.cameras = np.empty((len(self.fmt_list), dfcs_vipa.ROWS, dfcs_vipa.COLS))
        for i, num in enumerate(self.fmt_list, start=0):
            self.cameras[i] = self.read_cam_avg(num)
        # self.cameras[0] = 0.0

    def fit_pixel(self, x, y, init_range):
        xi, yi = x[init_range], y[init_range]
        a = curve_fit(linear, xi, yi)[0][0]
        res = linear(x, a)/y

        # corr = np.ones(y.size)
        # corr[:35] = np.linspace(0.65, 1.0, 35)
        # res = res*corr

        return splrep(y, res, k=1, task=0, s=0.0)[:2]

    @staticmethod
    def monotonize(t, p):
        idx = np.unique(np.maximum.accumulate(p), return_index=True)[1]

        return t[idx], p[idx]

    def fit(self, init_range):
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

    def cal_matrix(self, cam):
        log.info('calculating calibration matrix')
        ret = np.full(cam.shape, 1.0)
        for p, (knots, coeffs) in self.calibration.items():
            ret[p] = splev(cam[p], (knots, coeffs, 1), ext=3)

        return ret

    def single_pixel(self, row, col):
        """Return a calibrating function based on a single-pixel fit."""
        knots, coeffs = self.calibration[(row, col)]

        def cal_func(cam):
            return splev(cam, (knots, coeffs, 1), ext=3)

        return cal_func

    def __call__(self, cam):
        return self.cal_matrix(cam)
