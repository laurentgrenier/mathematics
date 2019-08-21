import numpy as np

# First we define the functions,
def f (x, y) :
    return np.exp(-(2*x*x + y*y - x*y) / 2)

def g (x, y) :
    return x*x + 3*(y+1)**2 - 1

# Next their derivatives,
def dfdx (x, y) :
    return 1/2 * (-4*x + y) * f(x, y)

def dfdy (x, y) :
    return 1/2 * (x - 2*y) * f(x, y)

def dgdx (x, y) :
    return 2 * x

def dgdy (x, y) :
    return 6 * y + 6

from scipy import optimize
-1.450

def DL (xyλ) :
    [x, y, λ] = xyλ
    return np.array([
            dfdx(x, y) - λ * dgdx(x, y),
            dfdy(x, y) - λ * dgdy(x, y),
            - g(x, y)
        ])


g = lambda  x: np.sqrt(1-x**2) / np.sqrt(3) - 1
dg = lambda x: -x / (np.sqrt(3) * np.sqrt(1 - x ** 2))

import pandas as pd

def newton_raphson(x, f, d_f):
    d = {"x": [x], "f(x)": [f(x)], "d_f(x)": [d_f(x)]}

    for i in range(0, 10):
        print(i)
        x = x - f(x) / d_f(x)
        d["x"].append(x)
        d["f(x)"].append(f(x))
        d["d_f(x)"].append(d_f(x))
        print(d)

    return pd.DataFrame(d, columns=['x', 'f(x)', 'd_f(x)'])

