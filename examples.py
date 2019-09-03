import xrd_analysis as xrd
import matplotlib.pyplot as pl
import numpy as np


# unimodal fitting
# GCB
# read raw data and baseline correct
data = xrd.correct_baseline(xrd.read_xye("test.xye"))

# constants
# lambda is from the machine settings, it is the Cu-K-a line wavelength in nm
lambda_ = 0.154
# these two are from the paper
Kc = 0.9
Ka = 1.84

# 002 peak
# ----------------------------------------------------------------------
# d002 parameter
# we 'cut out' the x-y range that we are interested in
# this defines a rectangle that encompasses the peak that we want to fit to
x_range = (19, 28)
y_range = (44, 500)
# fit peak - call fit_peak with the default gaussian peak function
p_002, v_002 = xrd.fit_peak(data, x_range, y_range)
# get theta_002 and its variance - these are two outputs of fit_peak
theta_002, d_theta_002 = p_002[1], v_002[1]
# this just calculates the min and max of theta based on the mean and variance
theta_002_min, theta_002_max = theta_002 - d_theta_002, theta_002 + d_theta_002
# calculate parameter range of d002 - call get_d002 on the min and max
d_002_min, d_002_max = xrd.get_d002(lambda_, theta_002_min), \
                       xrd.get_d002(lambda_, theta_002_max)
# report - print out in the console
print("\n")
print("GCB parameters:")
print("-------------------------")
# we report the mean and the confidence range
print("d002 is: {:.4f} +/- {:.4f} nm".format(
    0.5 * (d_002_min + d_002_max),
    d_002_max - d_002_min
))

# Lc parameter
# this needs the FWHM of the peak fit - that is calculated from the standard
# deviation (sigma) parameter of the fits
# we calculate the min-max range of the FWHM
B_002_min, B_002_max = xrd.get_FWHM(np.abs(p_002[2]) - v_002[2]), \
                       xrd.get_FWHM(np.abs(p_002[2]) + v_002[2])
# we calculate the Lc parameter based on the range
Lc_min, Lc_max = xrd.get_L(lambda_, Kc, B_002_max, theta_002_max), \
                 xrd.get_L(lambda_, Kc, B_002_min, theta_002_min)
# print out results, mean and confidence range
print("Lc is: {:.4f} +/- {:.4f} nm".format(
    0.5 * (Lc_min + Lc_max),
    Lc_max - Lc_min
))

# 100 peak
# ----------------------------------------------------------------------
# La parameter
# the same as above, but with a different peak
x_range = (43, 44)
y_range = (73, 500)
# fit peak
p_100, v_100 = xrd.fit_peak(data, x_range, y_range)
theta_100, d_theta_100 = p_100[1], v_100[1]
theta_100_min, theta_100_max = theta_100 - d_theta_100, theta_100 + d_theta_100
B_100_min, B_100_max =xrd. get_FWHM(np.abs(p_100[2]) - v_100[2]), \
                       xrd.get_FWHM(np.abs(p_100[2]) + v_100[2])
La_min, La_max = xrd.get_L(lambda_, Ka, B_100_max, theta_100_max), \
                 xrd.get_L(lambda_, Ka, B_100_min, theta_100_min)
print("La is: {:.4f} +/- {:.4f} nm".format(
    0.5 * (La_min + La_max),
    La_max - La_min
))

# plot results
pl.figure()
# plot baseline corrected data
pl.plot(data[:, 0], data[:, 1])
# plot fits
x_ = np.linspace(data[:, 0].min(), data[:, 0].max(), 1000)
pl.plot(x_, xrd.gaussian(x_, *p_002))
pl.plot(x_, xrd.gaussian(x_, *p_100))
pl.xlabel("2theta")
pl.title("GCB")


# multimodal fitting
# here we try to fit a gaussian mixture over the region where the unimodal
# 100 peak should be. this is because in this region we have multiple peaks
# overlapping and we want to separate out the graphite 100 peak.

# this is a range that encompasses three overlapping peaks that we want to
# separate
x_range = (37.5, 60)
y_range = (9.5, 200)
# we carefully estimate the initial values of the parameters
# mu's are easy - these are the approximate location of the modes along the
# 2*theta axis. the parameters are 3-tuples. the first element in the tuples
# corresponds to the Fe peak, the second one to the graphite 100 peak and the
# third to the flat peak to the right of the C 100 peak.
mu = (43.6, 43.0, 51.7)
# these are the spreads of the gaussian peaks - they are guesstimated based on
# which peak appears wider and narrower.
sigma = (0.3, 3.0, 4.0)
# the heights of the peaks are estimated and then divided by two to eyeball
# the height of the individual modes
a = (130 / 2, 60, 25 / 2)
# we pack our guesstimates into a tuple to pass to fit_peak
p0 = (*a, *mu, *sigma)
# call fit_peak with a trimodal gaussian peak function
p, v = xrd.fit_peak(
    data,
    x_range,
    y_range,
    p0=p0,
    peak_function=xrd.trimodal_gaussian
)

# plot
pl.figure()
# baseline corrected data
pl.plot(data[:, 0], data[:, 1])
# plot individual modes
for k in range(3):
    pl.plot(x_, xrd.gaussian(x_, p[k], p[k + 3], p[k + 6]))
# plot superposition of modes
pl.plot(x_, xrd.trimodal_gaussian(x_, *p))
pl.xlabel("2theta")
pl.title("GCB")

# recalculate La from the second mode which is supposedly the graphite 100 peak
# we need to find the parameters in the parameter vector p. theta is the fourth
# one, sigma is the seventh one. otherwise this process is the same as above.
theta_100, d_theta_100 = p[4], v[4]
theta_100_min, theta_100_max = theta_100 - d_theta_100, theta_100 + d_theta_100
B_100_min, B_100_max = xrd.get_FWHM(np.abs(p[7]) - v[7]), \
                       xrd.get_FWHM(np.abs(p[7]) + v[7])
La_min, La_max = xrd.get_L(lambda_, Ka, B_100_max, theta_100_max), \
                 xrd.get_L(lambda_, Ka, B_100_min, theta_100_min)
print("\n")
print("GCB, corrected p100 peak")
print("La is: {:.4f} +/- {:.4f} nm".format(
    0.5 * (La_min + La_max),
    La_max - La_min
))