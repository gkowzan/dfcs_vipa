"""Unit conversion."""
from scipy.constants import lambda2nu, nu2lambda, c, h, k


def nu2wn(nu):
    """Converts frequency to wavenumber in reciprocal centimeters."""
    return nu/c*1e-2


def wn2nu(wn):
    """Converts wavenumber in reciprocal centimeters to frequency."""
    return wn*c*1e2


def lambda2wn(lam):
    return nu2wn(lambda2nu(lam))


def wn2lambda(wn):
    return nu2lambda(wn2nu(wn))


def wn2joule(wn):
    return h*wn2nu(wn)


def joule2wn(E):
    return nu2wn(E/h)


def wn2x(wn, T):
    return wn2joule(wn)/k/T
