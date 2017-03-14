class DiffEq(object):
	def __init__(self, fx, fy, start_t, end_t, step_t, x0, y0):
		self.fx = fx
		self.fy = fy
		self.start_t = start_t
		self.end_t = end_t
		self.step_t = step_t
		self.x0 = x0
		self.y0 = y0

	def build_solution(self):
		x = []
		y = []
		x_y_data = []
		cur_t = self.start_t
		prev_x = self.x0
		prev_y = self.y0
		prev_t = cur_t
		while cur_t < self.end_t:
			cur_x = prev_x + (cur_t - prev_t) * self.fx(x = prev_x, y = prev_y, t = prev_t)
			cur_y = prev_y + (cur_t - prev_t) * self.fy(x = prev_x, y = prev_y, t = prev_t)
			x.append((cur_t, cur_x))
			y.append((cur_t, cur_y))
			x_y_data.append((cur_x, cur_y))
			prev_x = cur_x
			prev_y = cur_y
			prev_t = cur_t

			cur_t += self.step_t
		self.x = x
		self.y = y
		self.x_y_data = x_y_data


#x = var('x')
#y = var('y')
#t = var('t')
#g = Graphics()
#g += plot_vector_field((y, -x), (x,-1.1,1.1), (y,-1.1,1.1), color="blue")
#g += parametric_plot((cos(t),sin(t)),(t, 0, 2*pi),color="green")
#diff_eq = DiffEq(y, -x, 0, 2*pi, 0.01, 1, 0)
#diff_eq.build_solution()
#g += list_plot(diff_eq.x_y_data, plotjoined=True,color="red")
#g.save("diff_eq_solution.png")
#
#


x = var('x')
y = var('y')
t = var('t')
h = 0.3
log_error_points = {}
while h > 0.01:
	diff_eq = DiffEq(y, -x, 0, 2*pi, h, 1, 0)
	diff_eq.build_solution()

	print "for h={h}".format(h=h)
	for (real_func, approx_func, var_name) in ((cos(t), diff_eq.x, 'x'), (-sin(t), diff_eq.y, 'y')):
		max_norm = 0
		cur_t = 0
		for (t_, x_) in approx_func:
			if max_norm < abs(real_func(t_) - x_):
				max_norm = abs(real_func(t_) - x_)

		log_error_points.setdefault(var_name, []).append((-log(h), log(max_norm)))

	h -= 0.01

for var_name in ('x', 'y'):
	g = Graphics()
	g += list_plot(log_error_points[var_name], plotjoined=True, legend_label="{var_name}_error".format(var_name=var_name), gridlines=True)
	g.save("diff_eq_{var_name}_error.png".format(var_name=var_name))

#from sage.symbolic.integration.integral import definite_integral
#from sage.gsl.integration import numerical_integral


#
#rho = 6*t*(1 - t)
#S = 3*t + sin(3*t)
#z = 4*t + cos(t)
#x0 = 0
#y0 = 0
#T = 1
#step = 0.01
#beta = 0.03
#f_beta = beta * (S - x)
#
#diff_eq = DiffEq(
#	diff(z, t) * definite_integral(rho, t, y, 1),
#	f_beta,
#	0,
#	T,
#	0.01,
#	x0,
#	y0
#)
#
#diff_eq.build_solution()
#
##print diff_eq.x
##print diff_eq.y
#
#g = Graphics()
#g += list_plot(diff_eq.x, plotjoined=True, legend_label="x")
#g.show()
#
#
#g = Graphics()
#g += list_plot(list((S(t), x) for (t, x) in diff_eq.x), plotjoined=True, legend_label="x(s)")
#g.show()
#
#g = Graphics()
#g += list_plot(diff_eq.y, plotjoined=True, legend_label="y")
#g.show()
#
#g = Graphics()
#g += plot(S, (t, 0, T), legend_label="S")
#g.show()
#
#class ListToFunc(object):
#	def __init__(self, some_list):
#		self.some_dict = {}
#		for (x_, y_) in  some_list:
#			self.some_dict[floor(x_ / step) * step] = y_
#
#	def get_value(self, x):
#		new_value = floor(x / step) * step
#		if new_value == T:
#			new_value -= step
#		return self.some_dict[new_value]
#
#c2 = float(abs(ListToFunc(diff_eq.x).get_value(T) - S(T)) / S(T))
#print("c2", c2)
#
#x_func = ListToFunc(diff_eq.x)
#y_func = ListToFunc(diff_eq.y)
#
#def get_c2_subintegral(arg):
#	return float(definite_integral(t * rho(t), t, y, 1)(t = arg)(y = y_func.get_value(arg)) * (x_func.get_value(arg + step) - x_func.get_value(arg)) / step)
#
#c1 = (
#	1
#	- numerical_integral(get_c2_subintegral, 0, T, rule=1, max_points=20)[0] / (x_func.get_value(T) - x0)
#)
#print("c1", c1)
