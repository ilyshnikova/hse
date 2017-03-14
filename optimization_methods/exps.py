import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

def plot_levels(func, xrange=None, yrange=None, levels=None):
    """
    Plotting the contour lines of the function.

    Example:
    --------
    >> oracle = QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> plot_levels(oracle.func)
    """
    if xrange is None:
        xrange = [-6, 6]
    if yrange is None:
        yrange = [-5, 5]
    if levels is None:
        levels = [0, 0.25, 1, 4, 9, 16, 25]

    x = np.linspace(xrange[0], xrange[1], 100)
    y = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

#    CS = plt.contourf(X, Y, Z, levels=levels)
#    plt.clabel(CS, inline=1, fontsize=8)
    contour = plt.contour(X, Y, Z, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    cmap = plt.cm.get_cmap('YlGnBu')
    contour_filled = plt.contourf(X, Y, Z, levels, cmap=cmap, alpha=0.5)
    plt.colorbar(contour_filled)

    plt.grid()


def plot_trajectory(func, history, fit_axis=False, label=None):
    """
    Plotting the trajectory of a method.
    Use after plot_levels(...).

    Example:
    --------
    >> oracle = QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> [x_star, msg, history] = gradient_descent(oracle, np.array([3.0, 1.5], trace=True)
    >> plot_levels(oracle.func)
    >> plot_trajectory(oracle.func, history['x'])
    """
    x_values, y_values = zip(*history)
    x_values = np.array(x_values)
    y_values = np.array(y_values)


    plt.quiver(x_values[:-1], y_values[:-1], x_values[1:] - x_values[:-1], y_values[1:]-y_values[:-1], scale_units='xy', angles='xy', scale=1, linewidth=0.1)

    if fit_axis:
        xmax, ymax = np.max(x_values), np.max(y_values)
        COEF = 1.5
        xrange = [-xmax * COEF, xmax * COEF]
        yrange = [-ymax * COEF, ymax * COEF]
        plt.xlim(xrange)
        plt.ylim(yrange)


A = np.array([[10., 20.], [20., 5.]])
method = 'Wolfe'

oracle = oracles.QuadraticOracle(A, np.zeros(2))
[x_star, msg, history] = optimization.gradient_descent(
    oracle, np.array([3., 2.]),
    trace=True,
    line_search_options={
        'method': method,
        'c1': 1e-4,
        'c2': 0.3
    }
)

plot_levels(oracle.func)
plot_trajectory(oracle.func, history['x'], fit_axis=True)

plt.title('gradient descent with %s linear search, steps: %d\n for Quadratic function with\n A = %s' % (method, len(history['x']), str(A)))
plt.show()

print(np.linalg.cond(A))

