import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

X = np.loadtxt('X.csv', delimiter=',', unpack=True)
y = np.loadtxt('y.csv', delimiter=',', unpack=True)

gp = GaussianProcessRegressor(kernel=RBF(length_scale_bounds=(1e-10,1e10)), alpha=0.1**2)
gp.fit(X.reshape(-1, 1), y)

def sample_f(x):
    pred = gp.predict(x.reshape(-1, 1))
    return pred*np.random.normal(1, 1, len(x))