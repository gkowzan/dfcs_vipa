import pickle
import logging
import os
from pathlib import Path
import ctypes
import multiprocessing as mp
import h5py as h5
import numpy as np
from scipy.interpolate import splev, InterpolatedUnivariateSpline
from scipy.signal import convolve
from scipy.special import wofz
from dfcs_vipa import collect
from dfcs_vipa import grid
import dfcs_vipa.lineshape as ls
from dfcs_vipa.data.cmws import knots
from shed.experiment import find_maxima, denser, expand, remove_close, fwhm_est

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s: %(message)s',
)


def hdf5old2new_copy(path):
    done = False
    p2 = Path(path).with_suffix('.tmp')
    with h5.File(path, 'r') as f1:
        with h5.File(p2, 'w') as f2:
            if 'data' in f1:
                log.debug("all at once")
                f2.require_dataset("data",
                                   f1['data'].shape,
                                   maxshape=(None, 256, 320),
                                   dtype='u2', compression='lzf')
                f2["data"][...] = f1["data"][...]
                done = True
            elif '000001' in f1:
                log.debug("one by one")
                length = sum(1 for _ in f1.keys())
                f2.require_dataset("data",
                                   (length, 256, 320),
                                   maxshape=(None, 256, 320),
                                   dtype='u2', compression='lzf')
                # read in all data
                temp = np.empty((length, 256, 320), dtype='u2')
                for name in f1.keys():
                    num = int(name)
                    temp[num] = f1[name][...]
                f2['data'][...] = temp[...]
                done = True

    if done:
        p2.rename(path)


def dir_hdf5old2new_copy(path):
    log.debug("Entering '{}'".format(path))
    for entry in os.scandir(path):
        if entry.is_file() and entry.name.endswith('.hdf5'):
            log.debug("Converting '{}'".format(entry.name))
            hdf5old2new_copy(entry.path)


#########################################################################
# Fitting and work-flow                                                 #
#########################################################################
# grid helpers
def make_teeth_grid(arr, grid_points):
    teeth_spectrum = collect.collect(
        arr, grid.grid2fancy(grid_points), np.array([-1, 0, -1])
    )
    teeth_indices = remove_close(
        find_maxima(teeth_spectrum, window_len=5, order=6),
        10)

    grid_points_teeth = [grid_points[i]
                         for i in range(len(grid_points))
                         if i in teeth_indices]

    return teeth_indices, grid_points_teeth


def get_rio_pos_file(rio, rio_dc):
    return grid.get_rio_pos(
        collect.average_h5(rio, rio_dc))


def make_grid_file(grid_file, grid_dc):
    return grid.make_grid(
        collect.average_h5(grid_file, grid_dc
        ))


#########################################################################
# Frequency axis                                                        #
#########################################################################
def tooth_number(cw, frep, f0, fbeat=0.0, cavity_fsr=None):
    """Return tooth number closest to cw for given frep, f0.

    If cavity_fsr is not None, then we want to get the tooth number
    corresponding to transmitted comb tooth, which isn't same as the one
    with which we are beating.

    The returned comb tooth numer is always the closest one to the CW
    laser.  It can be lower freqency or higher frequency than the CW
    laser.

    """
    beaten_tooth = np.round((cw-f0-fbeat)/frep).astype(np.int)
    if cavity_fsr is not None:
        tooth_shift = np.round(fbeat/(cavity_fsr-frep)).astype(np.int)
    else:
        tooth_shift = 0
    log.info('CW tooth = {:d}, comb tooth = {:d}'.format(beaten_tooth,
                                                         tooth_shift))

    return beaten_tooth + tooth_shift


########################################################################
# Fitting                                                              #
########################################################################
def lorentz(x, x0, gamma, a):
    return a*1/np.pi*gamma/((x-x0)**2 + gamma**2)


def lorentz_diff(x, x0, gamma, a):
    """Return derivative of Lorentz function."""
    return -a*2*gamma*(x-x0)/np.pi/((x-x0)**2 + gamma**2)


def vp(x, x0, gam, dop):
    """The Voigt profile.

    Parameters
    ----------
    x : ndarray
        Frequency axis.
    x0 : float
        Line position with shift.
    gam : float
        Lorentzian half-width.
    dop : float
        Doppler half-width.
    """
    sigma = dop/np.sqrt(np.log(2))

    return wofz(((x - x0) + 1j*gam)/sigma)/sigma/np.sqrt(np.pi)



def voigt(x, x0, gamma, dop, a):
    """Return the Voigt profile (the real part)."""
    return a*np.real(vp(x, x0, gamma, dop))


def fit_mode(x, y, prof='lorentz', baseline=None, etalon=None,
             full_output=False):
    """Fit a single cavity mode.

    The initial guesses for the fit are calculated automatically.

    Parameters
    ----------
    x : ndarray
        Frequency data.
    y : ndarray
        Intensity data.
    prof : {'lorentz', 'voigt'}, optional
        Pick cavity mode model function.
    baseline : list, optional
        Initial guesses for polynomial baseline fitting in ascending
        order of polynomial terms, i.e. first one is the constant term.
    etalon : list, optional
        [amplitude, period, phase] of fitted etalon.
    full_output : bool, optional
        Decides if model should be returned also.

    Returns
    -------
    popt : ndarray
        Position, halfwidth, amplitude, intercept, slope.
    pstd : ndarray
        Fit standard deviations.
    model : ls.Model
        Callable pydfcs.lineshape.Model object.
    """
    if prof not in ('lorentz', 'voigt'):
        raise ValueError("'prof' should be one of: lorentz, voigt")
    try:
        model = ls.Model()
        if prof == 'lorentz':
            model.add_function(
                'mode',
                lorentz,
                # p0=(x[np.argmax(y)], 8e3, np.max(y)*np.pi*2*8e3)
                p0=(x[np.argmax(y)], fwhm_est(x, y)[0]/2, np.max(y)*np.pi*fwhm_est(x, y)[0])
            )
        elif prof == 'voigt':
            model.add_function(
                'mode',
                voigt,
                p0=(x[np.argmax(y)], fwhm_est(x, y)[0]/2, 1e3, np.max(y)*np.pi*fwhm_est(x, y)[0])
            )
        if baseline is not None:
            model.add_function(
                'baseline',
                ls.polynomial,
                baseline,
                nargs=len(baseline)
            )
        if etalon is not None:
            model.add_function(
                'etalon',
                ls.etalon,
                etalon,
            )
        model.fit_data(x, y)

        if full_output:
            return model.popt, np.sqrt(np.diag(model.pcov)), model
        else:
            return model.popt, np.sqrt(np.diag(model.pcov))
    except RuntimeError:
        # fitting failure
        return np.zeros(model.index), np.zeros(model.index)


def fit_modes(x, beat_spectrum, nshift=None, prof='lorentz', baseline=None,
              etalon=None):
    """Fit cavity modes from the whole measurement.

    Args:
    - x: relative frequency axis of each cavity mode,
    - beat_spectrum: 2D NumPy array, first axis - beat note values,
      second axis - comb teeth;
    - nshift: (n0, nCW) tuple with the teeth numbers of the first comb
      tooth in the spectrum and the comb tooth closest to the CW laser;
    - prof: fitting profile, either 'lorentz' or 'voigt'
    """
    if prof == 'lorentz':
        mode_fits = np.empty((beat_spectrum.shape[1], 2,
                              3 + (0 if baseline is None else len(baseline))
                              + (0 if etalon is None else 3)))
    elif prof == 'voigt':
        mode_fits = np.empty((beat_spectrum.shape[1], 2,
                              4 + (0 if baseline is None else len(baseline))
                              + (0 if etalon is None else 3)))
    for i in range(beat_spectrum.shape[1]):
        if nshift is not None:
            nCW, n0, nSkip = nshift
            xmode = x*(i*nSkip+n0)/nCW
        else:
            xmode = x
        mode_fits[i, :] = np.vstack(fit_mode(xmode, beat_spectrum[:, i],
                                             prof=prof, baseline=baseline,
                                             etalon=etalon))

    return mode_fits


#########################################################################
# Calibration                                                           #
#########################################################################
def init_calibrate(cal, output_base, shape):
    """Initialize the arrays for multiprocessing calibration."""
    global output_array
    global calibration

    calibration = cal
    output_array = np.ctypeslib.as_array(output_base).reshape(shape)


def worker_calibrate(i, arr):
    """Calibrate i-th array."""
    output_array[i] = arr*calibration(arr)


def calibrate_collection_mp(coll, cal, nproc=4):
    log.info('calibrating collection of size {:d}'.format(coll.shape[0]))
    global output_base
    output_base = mp.Array(
        ctypes.c_double, coll.size, lock=False
    )
    output = np.ctypeslib.as_array(output_base).reshape(coll.shape)
    chunksize = coll.shape[0]//nproc
    with mp.Pool(processes=nproc, initializer=init_calibrate,
                 initargs=(cal, output_base, coll.shape)) as pool:
        pool.starmap(worker_calibrate,
                     zip(
                         range(coll.shape[0]),
                         coll
                     ),
                     chunksize=chunksize)

    return output


def calibrate_collection(coll, cal):
    coll_cal = np.empty(coll.shape)
    for i in range(coll.shape[0]):
        coll_cal[i] = coll[i]*cal(coll[i])

    return coll_cal


def calibrate_collection_single(coll, cal):
    coll_cal = np.empty(coll.shape)
    for i in range(coll.shape[0]):
        for j in range(coll.shape[1]):
            coll_cal[i, j] = coll[i, j]*cal(coll[i, j])

    return coll_cal

            
def make_cal_func(coeffs):
    """Make a camera array calibration function."""
    def cal_func(arr):
        return splev(arr, (knots, coeffs, 1))

    return cal_func


def splev_wrap(x, tck):
    ret = splev(x, tck, ext=1)
    ret[ret == 0.0] = 1.0

    return ret


#########################################################################
# Data persistence                                                      #
#########################################################################
def save_h5(path, frep_fits, axis):
    with h5.File(path, 'w') as f:
        f.create_group('frep')
        for l in frep_fits.keys():
            f['frep'].create_dataset(
                str(l), data=frep_fits[l])

        f.create_group('axis')
        for l in axis.keys():
            f['axis'].create_dataset(
                str(l), data=axis[l])


if __name__ == '__main__':
    h5_dir = Path('/home/gkowzan/documents/nauka/fizyka/DFCS'
                  '/POMIARY/CCD/2017-05-29')
    dir_hdf5old2new_copy(str(h5_dir))
