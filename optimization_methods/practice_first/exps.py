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

#    CS = plt.contour(X, Y, Z, levels=levels, colors='k', linewidth=4.0)
#    plt.clabel(CS, inline=1, fontsize=8)

    contour = plt.contour(X, Y, Z, levels, colors='k')
    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
    cmap = plt.cm.get_cmap('YlGnBu')
    contour_filled = plt.contourf(X, Y, Z, levels, cmap=cmap, alpha=0.5)
    plt.colorbar(contour_filled)




#def plot_levels(func, xrange=None, yrange=None, levels=None):
#    """
#    Plotting the contour lines of the function.
#
#    Example:
#    --------
#    >> oracle = QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
#    >> plot_levels(oracle.func)
#    """
#    if xrange is None:
#        xrange = [-20, 20]
#    if yrange is None:
#        yrange = [-20, 20]
#    if levels is None:
#        levels = [0, 0.25, 1, 4, 9, 16, 25]
#
#    x = np.linspace(xrange[0], xrange[1], 100)
#    y = np.linspace(yrange[0], yrange[1], 100)
#    X, Y = np.meshgrid(x, y)
#    Z = np.zeros(X.shape)
#    for i in range(Z.shape[0]):
#        for j in range(Z.shape[1]):
#            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))
#
##    CS = plt.contourf(X, Y, Z, levels=levels)
##    plt.clabel(CS, inline=1, fontsize=8)
#    contour = plt.contour(X, Y, Z, levels, colors='k')
#    plt.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=12)
#    cmap = plt.cm.get_cmap('YlGnBu')
#    contour_filled = plt.contourf(X, Y, Z, levels, cmap=cmap, alpha=0.5)
#    plt.colorbar(contour_filled)
#
#    plt.grid()


#def plot_trajectory(func, history, fit_axis=False, label=None):
#    """
#    Plotting the trajectory of a method.
#    Use after plot_levels(...).
#
#    Example:
#    --------
#    >> oracle = QuadraticOracle(np.array([[1.0, 2.0], [2.0, 5.0]]), np.zeros(2))
#    >> [x_star, msg, history] = gradient_descent(oracle, np.array([3.0, 1.5], trace=True)
#    >> plot_levels(oracle.func)
#    >> plot_trajectory(oracle.func, history['x'])
#    """
#    x_values, y_values = zip(*history)
#    x_values = np.array(x_values)
#    y_values = np.array(y_values)
#
#    plt.quiver(x_values[:-1], y_values[:-1], x_values[1:] - x_values[:-1], y_values[1:]-y_values[:-1], scale_units='xy', angles='xy', scale=1, linewidth=0.1)
#
#    plt.plot(x_values, y_values, '-v', linewidth=2.0, ms=7.0, alpha=0.5, c='k', label=label)
#
#    if fit_axis:
#        xmax, ymax = np.max(x_values), np.max(y_values)
#        COEF = 2
#        xrange = [-xmax * COEF, xmax * COEF]
#        yrange = [-ymax * COEF, ymax * COEF]
#        plt.xlim(xrange)
#        plt.ylim(yrange)

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

	plot_levels(oracle.func, levels=levels)#, [x_min - size, x_max + size], [y_min - size, y_max + size])
	plot_trajectory(oracle.func, history['x'], fit_axis=False)

	plt.title(
		'%d steps' % (
			len(history['x']) - 1,
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
	A, b = load_svmlight_file('w8a.txt')#real-sim')#'gisette_scale')#'w8a.txt')
	logr = oracles.create_log_reg_oracle(A, b, 1/len(b))
	hists = dict()
	method = 'Wolfe'
	for opt, label in zip([optimization.gradient_descent, optimization.newton], ["Grad", "Newton"]):
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

	print("---------------------------------------- data ----------------------------------------")

	print("x: ", hists["Grad"]['time'])

	print("y: ", hists["Grad"]['func'])

	print("---------------------------------------- data ----------------------------------------")
	print("---------------------------------------- data ----------------------------------------")

	print("x: ", hists["Newton"]['time'])

	print("y: ", hists["Newton"]['func'])

	print("---------------------------------------- data ----------------------------------------")


	plt.plot(hists["Grad"]['time'], hists["Grad"]['func'], color='b', label="Gradient")
	plt.plot(hists["Newton"]['time'], hists["Newton"]['func'], color='g', label="Newton")
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


def exp_5():
	m, n = 500, 90
	A1 = np.random.randn(m, n)
	b1 = np.random.randn(m,)

	m, n = 500, 90
	A2 = np.random.randn(m, n)
	b2 = np.random.randn(m,)
	lamda1 = 1 / np.alen(b1)

#	A = np.array([[1., 2.], [2., 5.]])
#	b = np.array([0., 0.])
	lamda2 = 1 / np.alen(b2)
#	x_0 = np.array([3., 1.5])
	x_01 = np.ones(A1.shape[1])
	for i in range(A1.shape[1]):
		x_01[i] = i

	x_02 = np.ones(A2.shape[1])
	for i in range(A2.shape[1]):
		x_02[i] = i


	line_search_options = [
#		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.0001},
##		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.001},
#		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.01},
##		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.01},
#		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.1},
##		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.2},
#		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.3},
##		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.5},
#		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.7},
#		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.8},
##		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.85},
#		{'method': 'Wolfe', 'c1': 1e-3, 'c2': 0.9},
#		{'method': 'Armijo', 'c1': 0.001},
#		{'method': 'Armijo', 'c1': 0.01},
#		{'method': 'Armijo', 'c1': 0.1},
#		{'method': 'Armijo', 'c1': 0.2},
#		{'method': 'Armijo', 'c1': 0.9},
#		{'method': 'Armijo', 'c1': 0.93},
#		{'method': 'Armijo', 'c1': 0.95},

		{'method': 'Constant', 'c': 0.1},
		{'method': 'Constant','c': 0.5},
		{'method': 'Constant','c': 0.7},
		{'method': 'Constant','c': 0.9},
		{'method': 'Constant','c': 1},
		{'method': 'Constant','c': 3},
		{'method': 'Constant','c': 3.5},
		{'method': 'Constant','c': 10},
		{'method': 'Constant','c': 20},
		{'method': 'Constant','c': 30},
	]
	r = []
	r_k = []
	xs = []


	for i in line_search_options:
		print(i)
		if i['method'] == 'Wolfe':
			logr = oracles.create_log_reg_oracle(A1, b1, lamda1)

			df_0 = np.linalg.norm(logr.grad(x_01))**2
			x_star, msg, history = optimization.newton(logr, x_01, trace=True, line_search_options=i)
		else:
			logr = oracles.create_log_reg_oracle(A2, b2, lamda2)
			df_0 = np.linalg.norm(logr.grad(x_02))**2
			x_star, msg, history = optimization.newton(logr, x_02, trace=True, line_search_options=i)

#		import pdb; pdb.set_trace()
		r_k += [np.vectorize(lambda x: math.log(x**2/df_0, 10))(history['grad_norm'])]
		xs += [range(len(history['time']))]
		r += [np.vectorize(lambda x: abs(x - 0))(history['func'])]

	plt.xlabel('Iteration')
	plt.ylabel('ln(r_k)')

	for i in range(len(r)):
		print(xs[i])
		print(r_k[i])
		plt.plot(xs[i], r_k[i], label=(line_search_options[i]['method'] + ' ' + str(line_search_options[i]['c'])))
	plt.title("Constant")
	plt.legend()
	plt.show()

	plt.xlabel('Iteration')
	plt.ylabel('|f - f^*|')
	plt.title("Constant")
	for i in range(len(r)):
		plt.plot(xs[i], r[i], label=(line_search_options[i]['method'] + ' ' + ' ' + str(line_search_options[i]['c'])))

	plt.legend()
	plt.show()


def exp_4():
	np.random.seed(31415)
	m, n = 10000, 8000
	A = np.random.randn(m, n)
	b = np.sign(np.random.randn(m))
	lamda = 1 / np.alen(b)

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
	#	np.random.seed(31415)
#	m, n = 10000, 8000
##	m, n = 1000, 800
#	A = np.random.randn(m, n)
#	b = np.sign(np.random.randn(m))
#
#
#	us_oracle = oracles.create_log_reg_oracle(A, b, oracle_type='usual')
#	opt_oracle =oracles. create_log_reg_oracle(A, b, regcoef=None, oracle_type='optimized')
#
#	hists = dict()
#
#	method = 'Wolfe'
#	for (orac, name) in [(opt_oracle, 'optimized_oracle'),(us_oracle, 'usual_oracle')]:
#		x_star, msg, history = optimization.gradient_descent(
#			us_oracle,
#			np.zeros(A.shape[1]),
#			trace=True,
#			line_search_options={
#				'method': method,
#				'c1': 1e-4,
#				'c2': 0.3,
#				'c': 0.9
#			},
#			display=False
#		)
#
#		hists[name] = history
#		print(history['func'])
#
##	print("---------------------------------------- data ----------------------------------------")
##
##	print("x: ", history['time'])
##
##	print("y: ", history['func'])
##
##	print("---------------------------------------- data ----------------------------------------")
##
##	print(hists['usual_oracle']['func'])
##	print(hists['optimized_oracle']['func'])
#
#	plt.plot(hists["usual_oracle"]['time'], hists["usual_oracle"]['func'], color='b', label="usual_oracle")
#	plt.plot(hists["optimized_oracle"]['time'], hists["optimized_oracle"]['func'], color='g', label="optimized_oracle")
#	plt.xlabel('Seconds')
#	plt.ylabel('func val')
#	plt.legend(loc=2)
#	print("show")
#	plt.show()
#
#	plt.plot([i for i in range(len(hists['usual_oracle']['func']))], hists["usual_oracle"]['func'], color='b', label="usual_oracle", alpha=0.5)
#	plt.plot([i for i in range(len(hists['optimized_oracle']['func']))], hists["optimized_oracle"]['func'], color='g', label="optimized_oracle", alpha=0.5)
#	plt.xlabel('Iteration number')
#	plt.ylabel('func val')
#	plt.legend(loc=2)
#	print("show")
#	plt.show()



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
#third_exp()
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#exp_5()
#----------------------------------------------------------------------



A = [np.array([[2.,0.],[0.,1.]]), np.array([[60.,0.],[0.,1.]])]
x_0 = [np.array([1.,1.]), np.array([20.,30.])]


#A = np.array([[60.,0.],[0.,1.]]) # 66.6278484037
#A = np.array([[1.,0.],[0.,2.]]) # 2.0

#A = np.array([[1.,0.],[0.,20.]]) #
#print(np.linalg.cond(A))
#
#x_0 = np.array([22.,34.])
method = 'Constant'

for a in A:
	for x in x_0:
		first_exp(a, x, method)

