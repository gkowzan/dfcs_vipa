# python imports
import os
import io
from glob import glob
import inspect
from functools import lru_cache
from itertools import chain
from collections import OrderedDict
from contextlib import contextmanager
# scipy stack
import numpy as np
from pathlib import Path
import numpy.polynomial.polynomial as pol
from scipy.special import erfcx
import scipy.constants as C
from scipy.optimize import curve_fit
# my own
from shed.units import wn2nu, nu2wn
c = C.c
N_A = C.N_A
k = C.k
amu = C.value('atomic mass constant')
opj = os.path.join

########################################################################
# Classes and functions for fitting arbitrary spectra                  #
########################################################################
class ModelMixIn:
    """Defines methods common to Model and ModelCombine."""
    def find(self, name):
        """Returns function dictionary of function 'name'."""
        return [func for func in self.models if func['name'] == name][0]

    def fit_data(self, x, y, **kwargs):
        """Fits provided data to the model.

        Args:
        - x, y: fitted data
        - kwargs: kwargs passed to curve_fit.
        """
        self.popt, self.pcov = curve_fit(self, x, y, p0=self.p0,
                                         method='lm', **kwargs)


class ModelCombine(ModelMixIn):
    """Combines different Models with a common part.

    Evaluates different Models with different argument vectors, and
    (possibly) evaluates a common Model with each models' argument
    vector and adds the result to the model.

    The Models can share arguments' values during evaluation. For
    example, when fitting multiple spectral lines, one can constrain the
    concentration argument to have to same value in all line Models.

    The input of the ModelCombine is a 1D vector which is split
    according to the indices set by set_partition method.  The resulting
    1D vectors are then used to call different models.  The output
    vectors from different models are then concatenated into the output.

    The main use of the class is to fit a spectrum obtained from
    separate measurements with different baselines.
    """
    def __init__(self):
        self.partition = None
        self.models = []
        self.cargs = []
        self.doCommon = True
        self.index = 0
        self.popt = None
        self.pcov = None

    @contextmanager
    def baseline(self):
        """Version of object without common models."""
        self.doCommon = False
        yield self
        self.doCommon = True

    def eval_commons(self, x):
        """Evaluate common models on `x` with fitted coeffs."""
        if self.popt is None:
            raise ValueError("Have to perform a fit before calling this function.")
        x = np.asarray(x)
        ysim = np.zeros(x.size)
        for common in self.common_models:
            ysim += common['model'](
                x, *self.get_popt(common['name'])
            )

        return ysim

    @property
    def common_models(self):
        return [model for model in self.models if model['common']]

    @property
    def individual_models(self):
        return [model for model in self.models if not model['common']]

    def add_model(self, name, model, cargs=None, common=False, scale=False):
        """Add a model, provide the guesses through the model methods.

        Optionally, specify common arguments, i.e. arguments shared with
        other models and/or treat the whole model as common to all the
        other un-common models.

        'nargs' is the number of model parameters reduced by the number
        of common arguments.

        Parameters
        ----------
        name : str
            Unique name of the model
        model : `Model`
            Instance of `Model`.
        cargs : dict, optional
            A dict of {carg_name: arg_index}, where carg is the name of
            one of self.cargs elements and arg_index is the position of
            the argument in the argument list of the current model,
        common : bool, optional
            If the Model should be evaluated for each un-common Model's
            input vector and added to the result.
        scale : list or bool, optional
            Whether to rescale result of evaluating this model and all
            common models.  The scaling factor will be an additional
            (last) argument for this model in argument list and the
            initial guess will be the value of this parameter.
        """
        cargs = {} if cargs is None else cargs
        nargs = model.index - len(cargs)
        nargs += len(scale) if scale else 0
        self.models.append({
            'name': name,
            'index': self.index,
            'model': model,
            'nargs': nargs,
            'common': common,
            'cargs': cargs,
            'scale': scale
        })
        self.index += nargs
        self.update_cargs()

    def update_cargs(self):
        """Update common arguments' call list indices.

        Called after adding a model."""
        for i, carg in enumerate(self.cargs):
            carg['index'] = self.index + i

    def add_carg(self, name, p0=None):
        """Define an argument with a common value across models.

        The common arguments are always at the end of the call list.
        """
        self.cargs.append({
            'name': name,
            'index': self.index + len(self.cargs),
            'p0': p0
        })

    def find_carg(self, name):
        """Return the common argument named 'name'."""
        return [carg for carg in self.cargs if carg['name'] == name][0]

    def arg_list(self, dmodel, args):
        """Pick dmodel arguments from args list."""
        fargs = list(args[dmodel['index']:dmodel['index']+dmodel['nargs']])
        for carg, mind in sorted(dmodel['cargs'].items(),
                                 key=lambda x: x[1]):
            carg = self.find_carg(carg)
            ind = carg['index']
            fargs.insert(mind, args[ind])

        return fargs

    def update_guesses(self, dpopt):
        """Update initial guesses of models in dpopt.

        Args:
        - dpopt: dict with 'name'->args_list of models, whose guesses
          should be updated.
        """
        for name, popt in dpopt.items():
            model = self.find(name)['model']
            model.update_guesses(popt)

    def __call__(self, x, *args, **kwargs):
        if self.partition is None:
            raise ValueError("You have to set the 'partition' field"
                             " before evaluating the model.")
        xs = np.split(x, self.partition)
        ys = []
        for x, model in zip(xs, self.individual_models):
            fargs = self.arg_list(model, args)
            if not model['scale']:
                scale = 1.0
            else:
                scale_coeffs = model['scale']
                scale = polynomial(x, *fargs[-len(scale_coeffs):])
                fargs = fargs[:-len(scale_coeffs)]
            y = model['model'](x, *fargs)
            if self.doCommon:
                for common in self.common_models:
                    fargs = self.arg_list(common, args)
                    y += common['model'](x, *fargs)
            y *= scale
            ys.append(y)

        return np.hstack(ys)

    @property
    def p0(self):
        """Return the guess list of the model.

        We do not want to modify the original Model objects in any way,
        so we take a copy of the guess list and remove the arguments
        which are common with other Model objects.
        """
        p0 = []
        # collect guesses for model parameters and remove dummy guesses
        # corresponding to common arguments
        for model in self.models:
            # we don't want to modify the guess list in place!
            p0m = model['model'].p0[:]
            for mind in reversed(sorted(model['cargs'].values())):
                del p0m[mind]
            p0.extend(p0m)
            # add scaling factor guess
            if model['scale']:
                p0.extend(model['scale'])

        # collect guesses for common arguments
        p0.extend([carg['p0'] for carg in self.cargs])

        return p0

    def get_popt(self, name):
        """Return fitted parameters of function 'name'."""
        func = self.find(name)

        return self.arg_list(func, self.popt)

    def get_submodel_popt(self, name, subname):
        """Return fitted parameters of function 'subname' inside 'name'.

        Works only for individual models.
        """
        model = self.find(name)
        model_popt = self.get_popt(name)
        submodel = model['model'].find(subname)

        return model_popt[submodel['index']:submodel['index']+submodel['nargs']]

    def get_popts(self, common=True, individual=True):
        """Return all fitted parameters as a dict.

        The result is directly applicable as an argument to
        self.update_guesses.
        """
        dpopt = OrderedDict()
        if common:
            for dmodel in self.common_models:
                dpopt[dmodel['name']] = self.get_popt(dmodel['name'])
        if individual:
            for dmodel in self.individual_models:
                dpopt[dmodel['name']] = self.get_popt(dmodel['name'])

        return dpopt

    def get_stdev(self, name):
        """Return errors of fitted parameters of function 'name'."""
        func = self.find(name)
        stdevs = np.sqrt(np.diag(self.pcov))

        return self.arg_list(func, stdevs)

    def get_pcov(self, name):
        """Return covariance submatrix of fitted parameters of 'name'.

        Works properly only for individual models.
        """
        func = self.find(name)

        return self.pcov[func['index']:func['index']+func['nargs'],
                         func['index']:func['index']+func['nargs']]

    def get_submodel_pcov(self, name, subname):
        """Return covariance submatrix of function 'subname' inside 'name'.

        Works only for individual models.
        """
        model = self.find(name)
        submodel = model['model'].find(subname)
        pcov = self.get_pcov(name)
        
        return pcov[submodel['index']:submodel['index']+submodel['nargs'],
                    submodel['index']:submodel['index']+submodel['nargs']]

    def get_stdevs(self, common=True, individual=True):
        dpopt = OrderedDict()
        if common:
            for dmodel in self.common_models:
                dpopt[dmodel['name']] = self.get_stdev(dmodel['name'])
        if individual:
            for dmodel in self.individual_models:
                dpopt[dmodel['name']] = self.get_stdev(dmodel['name'])

        return dpopt


class Model(ModelMixIn):
    """Defines a model function as a sum of provided functions.

    Can be used to fit arbitrary data.
    """
    def __init__(self):
        self.models = []
        self.index = 0

    def add_function(self, name, func, p0=None, nargs=None):
        """Add a function to the model, optionally provide initial guess
        for fitting.

        Parameters
        ----------
        name : str
            Unique name for the function.
        func : callable
            Function or other callable object.
        p0 : tuple, optional
            Initial guesses for function arguments.
        nargs : int, optional
            Number of arguments to call the functio with, if it accepts
            arbitrary number of arguments.
        """
        try:
            self.find(name)
            raise ValueError("Function with name '{:s}' is already present "
                             "in the model.".format(name))
        except IndexError:
            if nargs is None:
                nargs = len(inspect.getfullargspec(func)[0]) - 1
            self.models.append({
                'name': name,
                'index': self.index,
                'func': func,
                'p0': p0,
                'nargs': nargs
            })
            self.index += nargs

    def add_guess(self, name, p0):
        """Add initial guess for function 'name'."""
        func = self.find(name)
        func['p0'] = p0

    def update_guesses(self, popt):
        """Update initial guesses from fitting results."""
        for dfunc in self.models:
            p0 = popt[dfunc['index']:dfunc['index']+dfunc['nargs']]
            dfunc['p0'] = p0

    def __call__(self, x, *args, **kwargs):
        exclude = kwargs.get('exclude', [])
        result = np.zeros(x.size)
        for func in self.models:
            if func['name'] in exclude:
                pass
            else:
                fargs = args[func['index']:func['index']+func['nargs']]
                result += func['func'](x, *fargs)

        return result

    @property
    def p0(self):
        p0 = [func['p0'] for func in self.models]
        if None in p0:
            raise ValueError("You have to provide guesses for all"
                             " functions before fitting.")
        p0 = list(chain.from_iterable(p0))

        return p0

    def get_popt(self, name):
        """Return fitted parameters of function 'name'."""
        func = self.find(name)
        return self.popt[func['index']:func['index']+func['nargs']]

    def get_stdev(self, name):
        """Return errors of fitted parameters of function 'name'."""
        func = self.find(name)
        stdevs = np.sqrt(np.diag(self.pcov))
        return stdevs[func['index']:func['index']+func['nargs']]

    def param_descs(self):
        """Return table describing the correspondence between popt
        and pcov values and provided functions.
        """
        desc = []
        for func in self.models:
            name = func['name']
            args = inspect.getfullargspec(func['func'])[0][1:]
            nameargs = ['%s: %s' % (name, arg) for arg in args]
            desc.extend(nameargs)

        return list(enumerate(desc))


def etalon(x, A, T, phi):
    return A*np.sin(2*np.pi/T*x + phi)


def am_etalon(x, A, M, Tc, Tm, phic, phim):
    return (M*A*np.sin(2*np.pi/Tm*x+phim)+A)*np.sin(2*np.pi/Tc*x + phic)


def etalon_shifted(x0):
    def func(x, A, T, phi):
        return etalon(x-x0, A, T, phi)

    return func


def polynomial(x, *coeffs):
    """Calculate polynomial values at x.

    Args:
    - x: n-dimensional numpy array,
    - coeffs: polynomial coefficients in ascending order in respect to the
      order of the polynomial term.
    """
    return pol.polyval(x, coeffs)


def polynomial_shifted(x0):
    def func(x, *coeffs):
        return pol.polyval(x-x0, coeffs)

    return func


def polynomial_shifted_scaled(x0, scale):
    def func(x, *coeffs):
        return pol.polyval((x-x0)*scale, coeffs)

    return func


def polynomial_shift_coeffs(x0, coeffs):
    """Get `coeffs` of a polynomial shifted by `x0`."""
    import sympy as sp
    x = sp.Symbol("x")

    x0 = -x0
    sp_polynomial = sp.poly(sum([x**i*coeffs[i] for i in range(len(coeffs))]))
    sp_polynomial = sp_polynomial.shift(x0)

    return np.array(sp_polynomial.all_coeffs(), dtype=np.float)[::-1]


def polynomial_scale_coeffs(scale, coeffs):
    """Get `coeffs` of a polynomial with arguments scaled by `scale` factor."""
    return np.array([scale**i*coeffs[i] for i in range(len(coeffs))])
