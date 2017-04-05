# -*-coding: utf-8 -*-

import numpy as np
from numpy.linalg import LinAlgError
import scipy as sp
from datetime import datetime
from collections import defaultdict

import oracles
import optimization
import plot_trajectory_2d

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
        xrange = [-20, 20]
    if yrange is None:
        yrange = [-20, 20]
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

    plt.plot(x_values, y_values, '-v', linewidth=2.0, ms=7.0, alpha=0.5, c='k', label=label)

    if fit_axis:
        xmax, ymax = np.max(x_values), np.max(y_values)
        COEF = 2
        xrange = [-xmax * COEF, xmax * COEF]
        yrange = [-ymax * COEF, ymax * COEF]
        plt.xlim(xrange)
        plt.ylim(yrange)



def first_exp(A, x_0, method, b=np.zeros(2), levels=None):
#	A = np.array([[10., 20.], [20., 5.]])

	oracle = oracles.QuadraticOracle(A, b)
	[x_star, msg, history] = optimization.gradient_descent(
	    oracle, x_0,
	    trace=True,
	    line_search_options={
	        'method': method,
	        'c1': 1e-4,
	        'c2': 0.3,
		'c': 0.1
	    }
	)

	if msg != 'success':
		print(msg)
		return

	x_values, y_values = zip(*history['x'])
	x_max = max(x_values)
	x_min = min(x_values)
	y_max = max(y_values)
	y_min = min(y_values)

	size = max(x_max - x_min, y_max - y_min)

	plot_levels(oracle.func, levels=levels)#, [x_min - size, x_max + size], [y_min - size, y_max + size])
	plot_trajectory(oracle.func, history['x'], fit_axis=False)

	plt.title(
		'gradient descent with %s linear search, steps: %d\n from point %s \nfor Quadratic function with\n A = %s' % (
			method,
			len(history['x']) - 1,
			str(x_0),
			str(A.toarray())
		)
	)
	plt.show()

#	print(np.linalg.cond(A.toarray()))


def create_matrix(n, k):
	d = np.random.uniform(1, k, size=n)
	d[0] = 1
	d[-1] = k
	return sp.sparse.spdiags(d, 0, d.size, d.size)



def second_exp():
	ns = [10, 100, 1000, 10000, 100000] # размерность пространства
	ks = [5, 10, 30, 50, 80, 100, 150, 200, 350, 500, 700, 850, 1000] # обусловленость задачи
	colors = ['b', 'g', 'r', 'c', 'y', 'k']
	c_i = 0

	fig1 = plt.figure()
	ax1 = fig1.add_subplot(111)

	for n in ns:
		print(n)
		res_steps = []
		for iteration in range(10):
			steps = []
			for k in ks:
				print(k)
				A = create_matrix(n, k)
				#print('cond: ', np.linalg.cond(A.toarray()))
				#print('k:', k)
#				oracle = oracles.QuadraticOracle(A, np.array([0. for i in range(n)]))
#				x_0 = np.array([1. for i in range(n)])

				oracle = oracles.QuadraticOracle(A, np.array([1. for i in range(n)]))
				x_0 = np.array([0. for i in range(n)])
				method = 'Wolfe'
	#			import pdb; pdb.set_trace()
				[x_star, msg, history] = optimization.gradient_descent(
				    oracle, x_0,
				    trace=True,
				    line_search_options={
					'method': method,
					'c1': 1e-4,
					'c2': 0.3,
					'c': 0.1
				    }
				)
				steps.append(len(history['func']))
			if (iteration == 0):
				res_steps = steps
			else:
				for i in range(len(res_steps)):
					res_steps[i] += steps[i]

			ax1.plot(ks, steps, alpha=0.3, c=colors[c_i], ls='--')

		for i in range(len(res_steps)):
			res_steps[i] /= 10

#		import pdb; pdb.set_trace()
		ax1.plot(ks, res_steps, c=colors[c_i], label='n=%d' % n)
#		plt.plot(ks, res_steps, linewidth=2.0, ms=7.0, alpha=0.5, c=colors[c_i], )
		c_i += 1

	print("show")
#	colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired
#	colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
#	for i,j in enumerate(ax1.lines):
#		j.set_color(colors[i])

	ax1.legend(loc=2)
	plt.show()


from sklearn.datasets import load_svmlight_file
def third_exp():
	data = load_svmlight_file('w8a.txt')



#------------------------------------------------------------------------
#for k in [3, 10, 50, 100, 400, 700 , 1000]:
#	A = create_matrix(2, k)
#	x_0 = np.array([1,2.])
#	first_exp(A, x_0, method)
#------------------------------------------------------------------------


#------------------------------------------------------------------------
#A = create_matrix(2, 3)
#x_0 = np.array([1,2.])
#
#method = 'Wolfe'
#first_exp(A, x_0, method, np.array([1.,1.]), levels=[0, 0.5, 1.5, 2, 4])
#
#
#method = 'Armijo'
#first_exp(A, x_0, method, np.array([1.,1.]), levels=[0, 0.5, 1.5, 2, 4])
#
#method = 'Constant'
#first_exp(A, x_0, method, np.array([1.,1.]), levels=[0, 0.5, 1.5, 2, 4])
#------------------------------------------------------------------------


#----------------------------------------------------------------------
#second_exp()
#------------------------------------------------------------------------

#import pdb; pdb.set_trace()
#np.linalg.cond(create_matrix(10, 5).toarray())

#
#method = 'Wolfe'
#
## большое число обучсловленности
#A = np.array([[1,1],[1,1]])
## начальная точка не влияет на скорость сходимости
#x_0 = np.array([2,2.])
##first_exp(A, x_0, method)
#
#x_0 = np.array([-1.,-2.])
#first_exp(A, x_0, method)
#
#
## маленькое число обусловленности
#A = np.array([[2,1],[1,1]])
#x_0 = np.array([1.,2.])
#first_exp(A, x_0, method, levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1, 1.5, 2, 4])
#



#----------------------------------------------------------------------
third_exp()
#----------------------------------------------------------------------
