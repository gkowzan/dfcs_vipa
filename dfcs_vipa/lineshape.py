# python imports
from typing import List
from pathlib import PurePath
import os
import io
from glob import glob
import inspect
import operator as op
from functools import lru_cache
from itertools import chain
from collections import OrderedDict
from contextlib import contextmanager
# scipy stack
import numpy as np
from pathlib import Path
import numpy.polynomial.polynomial as pol
from scipy.linalg import svd
from scipy.special import erfcx
import scipy.constants as C
from scipy.optimize import curve_fit
import scipy.optimize as opt
# my own
from dfcs_vipa.units import wn2nu, nu2wn
c = C.c
N_A = C.N_A
k = C.k
amu = C.value('atomic mass constant')
opj = os.path.join

########################################################################
# Classes and functions for fitting arbitrary spectra                  #
########################################################################
def leastsquares_pcov(fit_result):
    _, s, VT = svd(fit_result.jac, full_matrices=False)
    thres = np.finfo(float).eps*max(fit_result.jac.shape)*s[0]
    s = s[s>thres]
    VT = VT[:s.size]
    pcov = np.dot(VT.T/s**2, VT)

    return pcov


class ModelMixIn:
    """Defines methods common to Model and ModelCombine."""
    def find(self, name):
        """Returns function dictionary of function 'name'."""
        return [func for func in self.models if func['name'] == name][0]

    def fit_data(self, x: np.ndarray, y: np.ndarray, **kwargs):
        """Fits provided data to the model.

        Parameters
        ----------
        x : ndarray
            Independent variable.
        y : ndarray
            Dependent variable.
        kwargs : dict
            Dict of named arguments passed to `scipy.optimize.least_squares`.
        """
        fit_func = lambda args: self(x, *args)-y
        if 'p0' in kwargs:
            p0 = kwargs['p0']
            del kwargs['p0']
        else:
            p0 = self.p0
        self.fit_result = opt.least_squares(
            fit_func, p0, bounds=self.bounds, **kwargs)
        self.popt = self.fit_result.x
        self.pcov = leastsquares_pcov(self.fit_result)
        self.pstdev = np.sqrt(np.diag(self.pcov))

class ModelCombine(ModelMixIn):
    """Combines different Models with a common part.

    Evaluates different Models with different argument vectors, and
    (possibly) evaluates a common Model with each models' argument
    vector and adds the result to the model.

    The Models can share arguments' values during evaluation. For
    example, when fitting multiple spectral lines, one can constrain the
    concentration argument to have to same value in all line Models.
    The shared arguments should only be defined in the top-level
    ModelCombine instance.

    The input of the ModelCombine is a 1D vector which is split
    according to the indices set by set_partition method.  The resulting
    1D vectors are then used to call different models.  The output
    vectors from different models are then concatenated into the output.

    ModelCombine can contain other ModelCombine or Model instances.  The
    tree of models should end with Model instances containing a single
    function.

    The main use of the class is to fit a spectrum obtained from
    separate measurements with different baselines.
    """
    def __init__(self, dtype=np.float64, oper=op.add):
        self.partition = None
        self.models = []
        self.shargs = []
        self.doCommon = True
        self.popt = None
        self.pcov = None
        self.dtype = dtype
        self.oper = oper
        self.ratio = False

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
        return [model for model in self.models
                if not (model['common'] or model['is_scaling'])]

    def add_model(self, name, model, common=False, scale_pointer=None,
                  is_scaling=False):
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
        model : `Model` or `ModelCombine`
            Instance of `Model` or `ModelCombine`.
        common : bool, optional
            If the Model should be evaluated for each un-common Model's
            input vector and added to the result.
        scale_pointer : str, optional
            Name of the model that will scale this model.
        is_scaling : bool, optional
            Is this model a scaling model.
        """
        try:
            self.find(name)
            raise ValueError("Model with name '{:s}' is already present.".format(
                name))
        except IndexError:
            if isinstance(model, Model):
                nargs = model.index
            elif isinstance(model, ModelCombine):
                nargs = model.nargs()
            self.models.append({
                'name': name,
                'model_index': len(self.models),
                'model': model,
                'nargs': nargs,
                'common': common,
                'scale_pointer': scale_pointer,
                'is_scaling': is_scaling,
            })

    def add_sharg(self, name: str, model_paths: List, p0=None, lower_bound=-np.inf,
                  upper_bound=np.inf):
        """Define an argument with a common value across SOME models in
        the model tree.
        
        Shared arguments are always at the end of the call list.
        """
        try:
            self.find_sharg(name)
            raise ValueError("Model with name '{:s}' is already present.".format(
                name))
        except IndexError:
            self.shargs.append({
                'name': name,
                'model_paths': model_paths,
                'index': len(self.shargs),
                'p0': p0,
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
            })
        
    def find_sharg(self, name):
        return [sharg for sharg in self.shargs if sharg['name'] == name][0]

    def shargs_num(self, sub_name):
        """Return number of shared arguments for `submodel`."""
        nshargs = 0
        for sharg in self.shargs:
            submodel_names = [p.split('/')[0] for p in sharg['model_paths']]
            nshargs += sum((1 for name in submodel_names if name==sub_name),
                           0)

        return nshargs

    # def resolve_model(self, parts):
    #     model = self
    #     while parts:
    #         name = parts.pop(0)
    #         model = model.find(name)['model']

    #     return model

    def model_path_index(self, parts):
        """Convert submodel path to top-level argument index."""
        parts = list(parts)
        num = 0
        parent = self
        while parts:
            name = parts.pop(0)
            num += parent.model_start(name)
            parent = parent.find(name)['model']

        return num
    
    def arg_path_index(self, path):
        """Convert (sub)model arg path to top-level model argument index."""
        path_parts = path.split('/')
        num = int(path_parts[-1])
        # print(path.parts[:-1])
        num += self.model_path_index(path_parts[:-1])

        return num

    def model_start(self, name):
        """Index of the first argument of the model in arguments list.

        Parameters
        ----------
        name : str
            Name of the model in self.models list.
        """
        i = self.find(name)['model_index']
        if i == 0:
            return 0
        else:
            return self.model_start(self.models[i-1]['name'])\
                + self.model_nargs(self.models[i-1]['name'])

    def nargs(self):
        num = 0
        for model in self.models:
            if isinstance(model['model'], ModelCombine):
                num += model['model'].nargs()
            elif isinstance(model['model'], Model):
                num += model['model'].index
            else:
                raise ValueError('One of contained models is not an instance of Model or ModelCombine.')

        return num

    def model_nargs(self, name):
        model = self.find(name)
        nargs = model['nargs'] - self.shargs_num(model['name'])

        return nargs

    def nonshargs_num(self):
        """Return number of arguments that are not shared."""
        names = (model['name'] for model in self.models)

        return sum(self.model_nargs(name) for name in names)
    
    def arg_list(self, dmodel: dict, args: list):
        """Pick dmodel arguments from args list.

        Find number and indices of shargs in the `dmodel` call list.
        """
        nargs = self.model_nargs(dmodel['name'])
        #print('nargs', nargs, dmodel['name'])
        start = self.model_start(dmodel['name'])
        #print('start', start, dmodel['name'])
        fargs = list(args[start:start+nargs])
        #print('fargs', fargs, dmodel['name'])
        indices = []
        for sharg in self.shargs:
            for path in sharg['model_paths']:
                if dmodel['name'] == path.split('/')[0]:
                    indices.append((self.arg_path_index(path)-start,  # fargs index
                                    sharg['index']+self.nonshargs_num()))  # args index
        indices.sort(key=lambda x: x[0])
        for ifargs, iargs in indices:
            fargs.insert(ifargs, args[iargs])

        return fargs

    def __call__(self, x, *args, **kwargs):
        ret = self.calc(x, *args, **kwargs)
        if self.ratio:
            with self.baseline() as base:
                return ret/base.calc(x, *args, **kwargs)
        else:
            return ret
            
    def calc(self, x, *args, **kwargs):
        if self.partition is not None:
            xs = np.split(x, self.partition)
        else:
            xs = [x]

        ys = []
        for x, model in zip(xs, self.individual_models):
            # individual function
            fargs = self.arg_list(model, args)
            #print(model['name'])
            #print(fargs)
            y = model['model'](x, *fargs)
            # common functions
            if self.doCommon:
                for common in self.common_models:
                    cargs = self.arg_list(common, args)
                    y = self.oper(y, common['model'](x, *cargs))
            # scale function
            if model['scale_pointer'] is not None:
                scaler = self.find(model['scale_pointer'])
                sargs = self.arg_list(scaler, args)
                y = scaler['model'](x, y, *sargs)
            ys.append(y)

        if self.partition is not None:
            return np.hstack(ys)
        else:
            return ys[0]

    @property
    def p0(self):
        """Return the guess list of the model.

        We do not want to modify the original Model objects in any way,
        so we take a copy of the guess list and remove the arguments
        which are common with other Model objects.
        """
        p0 = []
        # collect guesses for model parameters and remove dummy guesses
        # corresponding to shared arguments
        for model in self.models:
            # we don't want to modify the guess list in place!
            p0m = model['model'].p0[:]
            start = self.model_start(model['name'])
            sharg_indices = []
            for sharg in self.shargs:
                for path in sharg['model_paths']:
                    if model['name'] == path.split('/')[0]:
                        sharg_indices.append(self.arg_path_index(path)-start)
            sharg_indices.sort(reverse=True)
            for i in sharg_indices:
                del p0m[i]
            p0.extend(p0m)

        # collect guesses for common arguments
        p0.extend([sharg['p0'] for sharg in self.shargs])

        return p0

    @p0.setter
    def p0(self, val):
        for model in self.models:
            p0model = self.arg_list(model, val)
            if isinstance(model['model'], Model):
                model['model'].update_guesses(p0model)
            elif isinstance(model['model'], ModelCombine):
                model['model'].p0 = p0model
        if self.shargs:
            val_num = len(val)
            shargs_num = len(self.shargs)
            for i in range(shargs_num):
                self.shargs[i]['p0'] = val[val_num-shargs_num+i]

    @property
    def bounds(self):
        lower_bounds, upper_bounds = [], []
        for model in self.models:
            lb, ub = model['model'].bounds[:]
            start = self.model_start(model['name'])
            sharg_indices = []
            for sharg in self.shargs:
                for path in sharg['model_paths']:
                    if model['name'] == path.split('/')[0]:
                        sharg_indices.append(self.arg_path_index(path)-start)
            sharg_indices.sort(reverse=True)
            for i in sharg_indices:
                del lb[i]; del ub[i]
            lower_bounds.extend(lb)
            upper_bounds.extend(ub)
        lower_bounds.extend([sharg['lower_bound'] for sharg in self.shargs])
        upper_bounds.extend([sharg['upper_bound'] for sharg in self.shargs])

        return (lower_bounds, upper_bounds)

    def list_model_paths(self, parents=''):
        paths = []
        for model in self.models:
            if isinstance(model['model'], Model):
                paths.append(str(PurePath(parents, model['name'])))
            elif isinstance(model['model'], ModelCombine):
                paths.extend(model['model'].list_model_paths(
                    str(PurePath(parents, model['name']))))

        return paths
    
    def get_popt(self, name):
        """Return fitted parameters of function 'name'."""
        parts = name.split('/')
        parent = self
        popt = self.popt[:]
        while parts:
            name = parts.pop(0)
            model = parent.find(name)
            popt = parent.arg_list(model, popt)
            parent = model['model']

        return popt

    def get_stdev(self, name):
        """Return errors of fitted parameters of function 'name'."""
        parts = name.split('/')
        parent = self
        stdev = self.pstdev[:]
        while parts:
            name = parts.pop(0)
            model = parent.find(name)
            stdev = parent.arg_list(model, stdev)
            parent = model['model']

        return stdev

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
    def __init__(self, dtype=np.float64, oper=op.add):
        self.models = []
        self.index = 0
        self.dtype = dtype
        self.oper = oper

    @classmethod
    def from_function(cls, name, func, p0=None, nargs=None,
                      lower_bound=None, upper_bound=None):
        model = cls()
        model.add_function(name, func, p0, nargs, lower_bound,
                           upper_bound)

        return model

    def add_function(self, name, func, p0=None, nargs=None,
                     lower_bound=None, upper_bound=None):
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
                'nargs': nargs,
                'lower_bound': lower_bound if lower_bound is not None\
                else nargs*[-np.inf],
                'upper_bound': upper_bound if upper_bound is not None\
                else nargs*[np.inf]
            })
            self.index += nargs

    def update_guesses(self, popt):
        """Update initial guesses from fitting results."""
        for dfunc in self.models:
            p0 = popt[dfunc['index']:dfunc['index']+dfunc['nargs']]
            dfunc['p0'] = p0

    def __call__(self, x, *args, **kwargs):
        exclude = kwargs.get('exclude', [])
        result = np.empty(x.size, dtype=self.dtype)
        result[:] = neutral(self.oper)
        for func in self.models:
            if func['name'] in exclude:
                pass
            else:
                fargs = args[func['index']:func['index']+func['nargs']]
                result = self.oper(result, func['func'](x, *fargs))
                # result += func['func'](x, *fargs)

        return result

    @property
    def p0(self):
        p0 = [func['p0'] for func in self.models]
        if None in p0:
            raise ValueError("You have to provide guesses for all"
                             " functions before fitting.")
        p0 = list(chain.from_iterable(p0))

        return p0

    @p0.setter
    def p0(self, val):
        istart = 0
        for func in self.models:
            func['p0'][:] = val[istart:istart+len(func['p0'])]
            istart = len(func['p0'])
    
    @property
    def bounds(self):
        lower_bounds = [func['lower_bound'] for func in self.models]
        upper_bounds = [func['upper_bound'] for func in self.models]

        return (list(chain.from_iterable(lower_bounds)),
                list(chain.from_iterable(upper_bounds)))
    
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


def neutral(oper):
    if oper is op.add or oper is op.iadd:
        return 0.0
    elif oper is op.mul or oper is op.imul:
        return 1.0

    
def etalon(x, A, T, phi):
    return A*np.sin(2*np.pi/T*x + phi)


def am_etalon(x, A, M, Tc, Tm, phic, phim):
    return (M*A*np.sin(2*np.pi/Tm*x+phim)+A)*np.sin(2*np.pi/Tc*x + phic)


def etalon_limited(xmin, xmax):
    def func(x, A, T, phi):
        if x<xmin or x>xmax:
            return 0.0
        else:
            return etalon(x, A, T, phi)

    return np.vectorize(func)


def etalon_shifted(x0):
    def func(x, A, T, phi):
        return etalon(x-x0, A, T, phi)

    return func


def polynomial(x, *coeffs):
    """Calculate polynomial values at x.

    Parameters
    ----------
    x : np.ndarray
        ND array of independent variable values.
    coeffs
        polynomial coefficients in ascending order with respect to the order of
        the polynomial term.
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
