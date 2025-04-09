import numpy as np
from scipy.interpolate import interp1d
import warnings

X = np.loadtxt('X.csv', delimiter=',', unpack=True)
y = np.loadtxt('y.csv', delimiter=',', unpack=True)

interp = interp1d(X, y, kind='cubic', fill_value='extrapolate')

def sample_f(x):
    return interp(x)*np.random.normal(1, 1, len(x))