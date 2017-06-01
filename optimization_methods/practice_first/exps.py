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
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
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

    contour = plt.contour(X, Y, Z, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    cmap = plt.cm.get_cmap('YlGnBu')
    contour_filled = plt.contourf(X, Y, Z, levels, cmap=cmap, alpha=0.5)
    plt.colorbar(contour_filled)

def plot_trajectory(func, history, fit_axis=False, label=None):
    """
    Plotting the trajectory of a method.
    Use after plot_levels(...).

    Example:
    --------
    >> oracle = oracles.QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
    >> [x_star, msg, history] = optimization.gradient_descent(oracle, np.array([3.0, 1.5], trace=True)
    >> plot_levels(oracle.func)
    >> plot_trajectory(oracle.func, history['x'])
    """
    x_values, y_values = zip(*history)
    plt.plot(x_values, y_values, marker='v', mec="k",
             c='r', label=label)

    # Tries to adapt axis-ranges for the trajectory:
    if fit_axis:
        xmax, ymax = np.max(x_values), np.max(y_values)
        COEF = 1.5
        xrange = [-xmax * COEF, xmax * COEF]
        yrange = [-ymax * COEF, ymax * COEF]
        plt.xlim(xrange)
        plt.ylim(yrange)


def first_exp(A, x_0, method, opt_method, b=np.zeros(2), levels=None):
	oracle = oracles.QuadraticOracle(A, b)
	[x_star, msg, history] = opt_method(
	    oracle, x_0,
	    trace=True,
	    line_search_options={
	        'method': method,
	        'c1': 1e-4,
	        'c2': 0.3,
		'c': 0.2
	    },
	)

	if msg != 'success':
		print(msg)

	x_values, y_values = zip(*history['x'])
	x_max = max(x_values)
	x_min = min(x_values)
	y_max = max(y_values)
	y_min = min(y_values)

	size = max(x_max - x_min, y_max - y_min)

	plot_levels(oracle.func, levels=levels)
	plot_trajectory(oracle.func, history['x'], fit_axis=False)

	plt.title(
		'%d steps' % (
			len(history['x']) - 1,
		)
	)
	plt.show()


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


	for n in ns:
		print(n)
		res_steps = []
		for iteration in range(10):
			steps = []
			for k in ks:
				print(k)
				A = create_matrix(n, k)
				oracle = oracles.QuadraticOracle(A, np.array([1. for i in range(n)]))
				x_0 = np.array([0. for i in range(n)])
				method = 'Wolfe'
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

			plt.plot(ks, steps, alpha=0.3, c=colors[c_i], ls='--')

		for i in range(len(res_steps)):
			res_steps[i] /= 10

		plt.ylabel('iteration number')
		plt.xlabel('condition')
		plt.plot(ks, res_steps, c=colors[c_i], label='n=%d' % n)
		c_i += 1

	print("show")

	plt.legend(loc=2)
	plt.show()


from sklearn.datasets import load_svmlight_file
def third_exp(dataset_path, opts):
	A, b = load_svmlight_file(dataset_path)#real-sim')#'gisette_scale')#'w8a.txt')
	logr = oracles.create_log_reg_oracle(A, b, 1/len(b))
	hists = dict()
	method = 'Wolfe'
	for opt, label in opts:
		x_star, msg, history = opt(
						logr,
						np.zeros(A.shape[1]),
						trace=True,
				    		line_search_options={
							'method': method,
							'c1': 1e-4,
							'c2': 0.3,
							'c': 0.9
				    		},
						display=True
					)
		hists[label] = history


	for opt, label in opts:
		plt.plot(hists["label"]['time'], hists["label"]['func'], label=label)
	plt.xlabel('Secs')
	plt.ylabel('func val')
	plt.legend(loc=2)
	print("show")
	plt.show()

	df_0 = np.linalg.norm(logr.grad(np.zeros(A.shape[1])))**2
	r_k_gd = np.vectorize(lambda x: math.log(np.linalg.norm(x**2)/df_0))(hists['Grad']['grad_norm'])
	r_k_new = np.vectorize(lambda x: math.log(np.linalg.norm(x**2)/df_0))(hists['Newton']['grad_norm'])
	plt.plot(hists["Grad"]['time'], r_k_gd, color='b')
	plt.plot(hists["Newton"]['time'], r_k_new, color='g')
	plt.xlabel('Secos')
	plt.ylabel('ln(r_k)')
	print("show")
	plt.show()


def exp_5(cur_method, opt_method):
	m, n = 500, 90
	A1 = np.random.randn(m, n)
	b1 = np.random.randn(m,)

	m, n = 500, 90
	A2 = np.random.randn(m, n)
	b2 = np.random.randn(m,)
	lamda1 = 1 / np.alen(b1)

	lamda2 = 1 / np.alen(b2)
	x_01 = np.ones(A1.shape[1])
	for i in range(A1.shape[1]):
		x_01[i] = i

	x_02 = np.ones(A2.shape[1])
	for i in range(A2.shape[1]):
		x_02[i] = i

	if cur_method == 'Wolfe':
		line_search_options = [
			{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.0001},
			{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.01},
			{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.1},
			{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.3},
			{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.7},
			{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.8},
			{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.9},
		]

	if cur_method == 'Armijo':
		line_search_options = [
			{'method': 'Armijo', 'c1': 0.001},
			{'method': 'Armijo', 'c1': 0.01},
			{'method': 'Armijo', 'c1': 0.1},
			{'method': 'Armijo', 'c1': 0.2},
			{'method': 'Armijo', 'c1': 0.9},
			{'method': 'Armijo', 'c1': 0.93},
			{'method': 'Armijo', 'c1': 0.95},
		]
	if cur_method == 'Constant':
		line_search_options = [
			{'method': 'Constant', 'c': 0.1},
			{'method': 'Constant','c': 0.5},
			{'method': 'Constant','c': 0.7},
			{'method': 'Constant','c': 0.9},
			{'method': 'Constant','c': 1},
			{'method': 'Constant','c': 3},
			{'method': 'Constant','c': 3.5},
		]
	r = []
	r_k = []
	xs = []


	for i in line_search_options:
		print(i)
		if i['method'] == 'Wolfe':
			logr = oracles.create_log_reg_oracle(A1, b1, lamda1)
			df_0 = np.linalg.norm(logr.grad(x_01))**2
			x_star, msg, history = opt_method(logr, x_01, trace=True, line_search_options=i)
		else:
			logr = oracles.create_log_reg_oracle(A2, b2, lamda2)
			df_0 = np.linalg.norm(logr.grad(x_02))**2
			x_star, msg, history = opt_method(logr, x_02, trace=True, line_search_options=i)

		r_k += [np.vectorize(lambda x: math.log(x**2/df_0, 10))(history['grad_norm'])]
		xs += [range(len(history['time']))]
		r += [np.vectorize(lambda x: abs(x - 0))(history['func'])]

	plt.xlabel('Iteration')
	plt.ylabel('ln(r_k)')

	for i in range(len(r)):

		print(xs[i])
		print(r_k[i])
		plt.plot(xs[i], r_k[i], label=(line_search_options[i]['method'] + ' ' + str(line_search_options[i])))
	plt.title(cur_method)
	plt.legend()
	plt.show()

	plt.xlabel('Iteration')
	plt.ylabel('|f - f^*|')
	plt.title(cur_method)
	for i in range(len(r)):
		plt.plot(xs[i], r[i], label=(line_search_options[i]['method'] + ' ' + ' ' + str(line_search_options[i])))

	plt.legend()
	plt.show()


def exp_4():
	m, n = 10000, 8000
	A = np.random.randn(m, n)
	b = np.sign(np.random.randn(m))
	lamda = 1 / np.alen(b)

	hists = dict()

	for opt in ["optimized", "usual"]:
	    logr = oracles.create_log_reg_oracle(A, b, lamda, opt)
	    x_star, msg, history = optimization.gradient_descent(logr, np.zeros(A.shape[1]), trace=True)
	    print(opt)
	    hists[opt] = history
	plt.clf()
	plt.plot(range(len(hists["usual"]['time'])), hists["usual"]['func'], label='usual', color='blue', linestyle='--', alpha=0.7)
	plt.plot(range(len(hists["optimized"]['time'])), hists["optimized"]['func'], label='optimized', color='red', linestyle=':', alpha=0.7)
	plt.xlabel('Iteration')
	plt.ylabel('F(x)')
	plt.legend(loc=2)
	plt.show()
	plt.clf()

	plt.plot(hists["usual"]['time'], hists["usual"]['func'], label='usual', color='blue')
	plt.plot(hists["optimized"]['time'], hists["optimized"]['func'], label='optimized', color='green')
	plt.xlabel('Seconds')
	plt.ylabel('F(x)')
	plt.legend(loc=2)
	plt.show()
	plt.clf()

	df_0 = np.linalg.norm(logr.grad(np.zeros(A.shape[1])))**2
	r_k_gd = np.vectorize(lambda x: math.log(np.linalg.norm(x**2)/df_0))(hists['usual']['grad_norm'])
	r_k_new = np.vectorize(lambda x: math.log(np.linalg.norm(x**2)/df_0))(hists['optimized']['grad_norm'])
	plt.plot(hists["usual"]['time'], r_k_gd, label='usual', color='blue')
	plt.plot(hists["optimized"]['time'], r_k_new, label='optimized', color='green')
	plt.xlabel('Seconds')
	plt.ylabel('ln(r_k)')
	plt.legend(loc=2)
	plt.show()



np.random.seed(31415)
#----------------------------------------------------------------------
#second_exp()
#------------------------------------------------------------------------
#----------------------------------------------------------------------
#path = ''
#third_exp(path)
#----------------------------------------------------------------------
#---------------------------------------------------------------------
#for i in ['Constant']:
#	for m in [optimization.newton, optimization.gradient_descent]:
#		exp_5(i, m)
#----------------------------------------------------------------------
## first exp
#----------------------------------------------------------------------
A = [np.array([[2.,0.],[0.,1.]]), np.array([[60.,0.],[0.,1.]])]
x_0 = [np.array([1.,1.]), np.array([20.,30.])]

method = 'Wolfe'

for a in A:
	for x in x_0:
		first_exp(a, x, method)
#---------------------------------------------------------------------
#exp_4()
