
# This file was *autogenerated* from the file homework2.sage
from sage.all_cmdline import *   # import sage library

_sage_const_3 = Integer(3); _sage_const_2 = Integer(2); _sage_const_1 = Integer(1); _sage_const_0 = Integer(0); _sage_const_6 = Integer(6); _sage_const_0p3 = RealNumber('0.3'); _sage_const_1p = RealNumber('1.'); _sage_const_2p = RealNumber('2.'); _sage_const_0p05 = RealNumber('0.05'); _sage_const_0p000001 = RealNumber('0.000001')
def get_modules_sum(numbers):
	return sum(abs(x) for x in numbers)

def get_line_modules_sum(cur_matrix):
	for line in cur_matrix:
		yield get_modules_sum(line)

def get_infty_norm(cur_matrix):
	return max(get_line_modules_sum(cur_matrix))

def get_conditionality_number(cur_matrix):
	return get_infty_norm(cur_matrix) * get_infty_norm(cur_matrix**(-_sage_const_1 ))

def elementary_transformation(cur_matrix, coefs, i, j, la):
	cur_matrix[i] -= cur_matrix[j] * float(la)
	coefs[i] -= coefs[j] * float(la)

def swap_rows(cur_matrix, coefs, i, j):
	(cur_matrix[i], cur_matrix[j]) = (cur_matrix[j], cur_matrix[i])
	(coefs[i], coefs[j]) = (coefs[j], coefs[i])

def gauss_solve(cur_matrix, coefs):
	for start_row_index in xrange(cur_matrix.nrows()):
		max_row_index = None
		max_col_value = None
		for row_index in xrange(start_row_index, cur_matrix.nrows()):
			if max_col_value == None or max_col_value < abs(cur_matrix[row_index][start_row_index]):
				max_col_value = abs(cur_matrix[row_index][start_row_index])
				max_row_index = row_index

		swap_rows(cur_matrix, coefs, start_row_index, max_row_index)

		for row_index in xrange(start_row_index + _sage_const_1 , cur_matrix.nrows()):
			elementary_transformation(
				cur_matrix,
				coefs,
				row_index,
				start_row_index,
				cur_matrix[row_index][start_row_index]
					/ cur_matrix[start_row_index][start_row_index]
			)

	for start_row_index in xrange(cur_matrix.nrows() - _sage_const_1 , -_sage_const_1 , -_sage_const_1 ):
		for row_index in xrange(start_row_index - _sage_const_1 , -_sage_const_1 , -_sage_const_1 ):
			elementary_transformation(
				cur_matrix,
				coefs,
				row_index,
				start_row_index,
				cur_matrix[row_index][start_row_index]
					/ cur_matrix[start_row_index][start_row_index]
			)

	for row_index in xrange(cur_matrix.nrows()):
		yield coefs[row_index] / cur_matrix[row_index][row_index]

def make_column(vector):
	return matrix([[x] for x in vector])

def calc_error_vector(cur_matrix, coefs, solution):
	return make_column(coefs) - cur_matrix * make_column(solution)

def calc_error(cur_matrix, coefs, solution):
	return max(line[_sage_const_0 ] for line in calc_error_vector(cur_matrix, coefs, solution))

'''
A = matrix([[1,2,3],[2.0001,3.999,6],[15,3,6]])
B = matrix(QQ, 8, 8, lambda i, j: 1./(i + j + 1))
C = matrix([[float(10^6),float(2)],[float(10^13),float(2)]])

rows = [["det","matrix_norm","conditionality_number","error"]]
for cur_matrix in (A, B, C):
	infty_norm = get_infty_norm(cur_matrix)
	conditionality_number = get_conditionality_number(cur_matrix)
	b = list(1 for i in xrange(cur_matrix.nrows()))
	solution = list(gauss_solve(cur_matrix, b))
	error = calc_error(cur_matrix, b, solution)
	rows.append([
		float(det(cur_matrix)),
		float(infty_norm),
		float(conditionality_number),
		error,
	])
print table(rows)
'''

def get_segments(a, b, step):
	x = a
	while (x + step <= b):
		yield (x, x + step)
		x += step
	if x < b:
		yield (x, b)

class CubeSpline(object):
	def __init__(self, a, b, c, d, x1):
		self.a = a
		self.b = b
		self.c = c
		self.d = d
		self.x1 = x1

	def get_value(self, x):
		return self.a + self.b * (x - self.x1) + self.c/_sage_const_2  * (x - self.x1)**_sage_const_2  + self.d / _sage_const_6  * (x - self.x1) ** _sage_const_3 

class SplineConvex(object):
	def __init__(self, func, a, b, step):
		self.splines = []
		self.segments = list(get_segments(a, b, step))
		n = len(self.segments)

		equations = [
			[_sage_const_1 ] + [_sage_const_0 ] * n,
			[_sage_const_0 ] * n + [_sage_const_1 ],
		]
		b = [
			_sage_const_0 ,
			_sage_const_0 ,
		]

		for i in xrange(_sage_const_1 , n):
			(x_i_prev, x_i) = self.segments[i - _sage_const_1 ]
			(x_i, x_i_post) = self.segments[i]
			f_i = func(x_i)
			f_i_post = func(x_i_post)
			f_i_prev = func(x_i_prev)
			h_i = x_i - x_i_prev
			h_i_post = x_i_post - x_i
			equations.append(
				[_sage_const_0 ] * (i - _sage_const_1 ) + [h_i, _sage_const_2  * (h_i + h_i_post), h_i_post] + [_sage_const_0 ] * (n - i - _sage_const_1 ),
			)
			b.append(_sage_const_6  * ((f_i_post - f_i) / h_i_post - (f_i - f_i_prev) / h_i))

		c_coefs = list(gauss_solve(matrix(equations), b))

		for i in xrange(_sage_const_1 , n + _sage_const_1 ):
			c_i = c_coefs[i]
			c_i_prev = c_coefs[i - _sage_const_1 ]
			(x_i_prev, x_i) = self.segments[i - _sage_const_1 ]
			f_i = func(x_i)
			f_i_prev = func(x_i_prev)
			h_i = x_i - x_i_prev
			a_i = f_i
			d_i = (c_i - c_i_prev) / h_i
			b_i = (f_i - f_i_prev) / h_i + h_i * (_sage_const_2  * c_i + c_i_prev) / _sage_const_6 
			self.splines.append(CubeSpline(a_i, b_i, c_i, d_i, x_i))


	def get_value(self, x):
		for index in xrange(len(self.segments)):
			(x1, x2) = self.segments[index]
			spline = self.splines[index]
			if x1 <= x and x <= x2:
				return spline.get_value(x)



for (func, a, b, func_name) in (
	(x**_sage_const_3 , _sage_const_1p , _sage_const_2p , "x_cubed"),
	(sgn(x), -_sage_const_1p , _sage_const_1p , "sign_x"),
	(sin(_sage_const_1 /x), _sage_const_0p000001 , _sage_const_1p , "sin_reverse_x"),
):

	g = Graphics()
	g += plot(func, (x, a, b), color='red', legend_label=func_name)
	for (step, color) in ((_sage_const_0p3 , 'blue'), (_sage_const_0p05 , 'green')):
		spline_convex = SplineConvex(func, a, b, step)
		g += plot(
			lambda t: spline_convex.get_value(t),
			(x, a, b), color=color,
			legend_label="{func_name}_step_{step:.2f}".format(func_name=func_name, step=float(step))
		)

	g.save("{func_name}_splines_image.png".format(func_name=func_name))

	g = Graphics()
	step = _sage_const_0p05 
	spline_convex = SplineConvex(func, a, b, step)
	g += plot(
		lambda t: func(t) - spline_convex.get_value(t),
		(x, a, b), color=color,
		legend_label="{func_name}_interpolation_error".format(func_name=func_name)
	)
	g.save("{func_name}_interpolation_error.png".format(func_name=func_name))

