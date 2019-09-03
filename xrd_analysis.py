# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:51:47 2019

A script for processing XRD spectra. Does baseline correction and peak fitting.

Baseline correction is done by Asymmetric Least Squares fitting. This method
is not rigorous. I will implement an exponential decay model in the future. For
now it works well for estimating La, Lc and d002.

How to use: read the docstrings of the individual functions to understand them.
Otherwise load the script in a Python IDE (Spyder or Visual Studio Code are the
best free ones), move your cursor to a code cell defined by a block with #%% on
the top, and press ctrl + enter. You should run the first (topmost) block
first. This contains function definitions and imports.

Requirements:
    
    python 3.6 +
    matplotlib
    numpy
    scipy

If a package is missing, you will get an "unknown module..." error or similar.
In this case, do 'conda install <package name>' or 'pip install <package name>'
in your command line. I recommend using Anaconda.

@author: Pal
"""

import matplotlib.pyplot as pl
import numpy as np

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit

from typing import Any, Callable, Optional, Tuple


def fit_baseline(
        y: np.ndarray,
        l: float = 1e7,
        p: float = 1e-4,
        n_iter: int = 100
) -> np.ndarray:
    """ Fits baseline by asymmetric least squares fitting.
    Reference: Eilers and Boelens: Baseline Correction with Asymmetric Least 
    Squares Smoothing
    
    :param y: The y values of the signal that you want to process.
    :type y: np.ndarray (1-D)
    :param l: The smoothness parameter.
    :type l: float
    :param p: The asymmetry parameter.
    :type p: float
    :param n_iter: The number of iterations.
    :type n_iter: int
    :returns: The baseline.
    :rtype: np.ndarray (1-D, same length as y)
    
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + l * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def correct_baseline(
        data: np.ndarray,
        l: float = 1e7,
        p: float = 1e-4,
        n_iter: int = 100
) -> np.ndarray:
    """ Subtracts the baseline from the y column of a data array.
    
    :param data: An n X 2 numpy array where the first column is the x
        coordinates (2 * theta) and the second column is the y values (the
        spectrum intensities). n is the number of datapoints.
    :type data: np.ndarray (size n X 2)
    :param l: The smoothness parameter.
    :type l: float
    :param p: The asymmetry parameter.
    :type p: float
    :param n_iter: The number of iterations.
    :type n_iter: int
    :returns: The baseline-corrected data array.
    :rtype: np.ndarray (2-D, same size as data)
    
    """
    data_ = data.copy()
    data_[:, 1] = data_[:, 1] - fit_baseline(data_[:, 1])
    return data_


def read_xye(path: str, delimiter: str = " ") -> np.ndarray:
    """ Reads a .xye file. Returns an n X 3 numpy array 'a' where a[:, 0] is 
    theta, a[:, 1] is the signal and a[:, 2] is something else.
    
    :param path: The full path to the .xye file.
    :type path: str
    :param delimiter: The delimiter in the .xye file.
    :type delimiter: str
    :returns: The read data.
    :rtype: np.ndarray (size n X 3)
    
    """
    return np.genfromtxt(path, delimiter=delimiter, skip_header=1)


def gaussian(
        x: np.ndarray,
        a: float,
        mu: float,
        sigma: float
) -> np.ndarray:
    """ A parametric Gaussian function. Used to fit peaks to baseline-corrected
    XRD spectra.
    
    :param x: The x coordinates.
    :type x: np.ndarray
    :param a: The height of the Gaussian.
    :type a: float
    :param mu: The mean of the Gaussian.
    :type mu: float
    :param sigma: The standard deviation of the Gaussian.
    :type sigma: float
    :returns: The y values of the Gaussian evaluated at x.
    :rtype: np.ndarray
    
    """
    return a * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))


def multimodal_gaussian(
        x: np.ndarray,
        a: Tuple[float, ...],
        mu: Tuple[float, ...],
        sigma: Tuple[float, ...]
) -> np.ndarray:
    """ A parametric multimodal Gaussian function. Used to fit peaks to 
    baseline-corrected spectra. Combinations of parameters are passed as
    tuples. The length of each tuple must be the same. The number of modes
    will be equal to the length of the tuples.
    
    :param x: The x coordinates.
    :type x: np.ndarray
    :param a: The heights of the Gaussians.
    :type a: Tuple[float, ...]
    :param mu: The means of the Gaussians.
    :type mu: Tuple[float, ...]
    :param sigma: The standard deviations of the Gaussians.
    :type sigma: Tuple[float, ...]
    :returns: The y values of the Gaussian evaluated at x.
    :rtype: np.ndarray
    
    """
    assert len(a) == len(mu) == len(sigma), \
        "The lengths of a, mu and sigma must be the same."

    y = np.zeros(x.shape)

    for a_, mu_, sigma_ in zip(a, mu, sigma):
        y = y + gaussian(x, a_, mu_, sigma_)

    return y


def trimodal_gaussian(x, a1, a2, a3, mu1, mu2, mu3, sigma1, sigma2, sigma3):
    """ Just a wrapper to pass to fit_peak. """
    return multimodal_gaussian(
        x,
        (a1, a2, a3),
        (mu1, mu2, mu3),
        (sigma1, sigma2, sigma3)
    )


def fit_peak(
        data: np.ndarray,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        p0: Optional[Tuple[Any, ...]] = None,
        peak_function: Callable = gaussian
) -> Tuple[Any, Any]:
    """ Fits a parametric peak function to a part of a 1-D spectrum. Returns 
    the fitted parameters and their confidence range.
    
    :param data: The data. A numpy array, size n X 2. First column is x, second
        columns is y. n is the number of data points.
    :type data: np.ndarray (size n X 2)
    :param x_range: The x range of the spectrum part (inclusive).
    :type x_range: Tuple[float, float] (min_x, max_x)
    :param y_range: The y range of the spectrum part (inclusive).
    :type y_range: Tuple[float, float] (min_y, max_y)
    :param p0: The initial value for the parameters. If None, assumes a
        Gaussian peak function.
    :type p0: Optional[Tuple[Any, ...]] (length equals number of parameters)
    :param peak_function: The parametric function to fit.
    :type peak_function: Callable
    
    """
    x = data[:, 0]
    y = data[:, 1]
    x_inds = np.logical_and(x >= x_range[0], x <= x_range[1])
    y_inds = np.logical_and(y >= y_range[0], y <= y_range[1])
    inds = np.logical_and(x_inds, y_inds)
    x, y = x[inds], y[inds]

    if p0 is None:
        mu = np.sum(x * y) / np.sum(y)
        sigma = np.sum(y * (x - mu) ** 2) / np.sum(y)
        a = np.max(y)
        p0 = [a, mu, sigma]

    p_opt, p_cov = curve_fit(peak_function, x, y, p0=p0)

    p_conf = 1.96 * np.sqrt(np.diag(p_cov))
    return p_opt, p_conf


def get_FWHM(sigma: float) -> float:
    """ Gets full width at half maximum of a Gaussian.
    
    :param sigma: Standard deviation of Gaussina.
    :type sigma: float
    :returns: The FWHM.
    :rtype: float
    
    """
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def get_d002(
        lambda_: float,
        theta_002: float
) -> float:
    """ Calculates d002 from theta_002 using the Bragg equation.
    
    :param lambda_: Wavelength of the X-ray beam.
    :type lambda_: float
    :param theta_002: The angle (2theta) of the 002 peak.
    :type theta_002: float
    :returns: The spacing in the unit of lambda_.
    :rtype: float
    
    """
    return lambda_ / (2 * np.sin(np.deg2rad(theta_002 / 2)))


def get_L(
        lambda_: float,
        K: float,
        B: float,
        theta: float
) -> float:
    """ Gets characteristic stack dimension.
    
    :param lambda_: The wavelength of the X-ray beam.
    :type lambda_: float
    :param K: Constant.
    :type K: float
    :param B: FWHM (del(2theta)) of the characteristic peak.
    :type B: float
    :param theta: The position of the peak along the 2*theta axis.
    :type theta: float
    :returns: The characteristic length in the unit of lambda_.
    :rtype: float
    
    """
    return K * lambda_ / (np.deg2rad(B) * np.cos(np.deg2rad(theta / 2)))
