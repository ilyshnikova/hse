def get_segments(a, b, step):
	x = a
	while (x + step <= b):
		yield (x, x + step)
		x += step
	if x < b:
		yield (x, b)

def trapesoid_integrate(func, a, b, step):
	integral = 0
	for (x, y) in get_segments(a, b, step):
		integral += (func(x) + func(y)) / 2 * (y - x)

	return integral

#from sage.symbolic.integration.integral import definite_integral
#
#for (func, a, b) in (
#	(x^3, 1, 2),
#	(sgn(x), -1, 1),
#	(sin(1/x), 0.000001, 1),
#):
#	print "Func {func} on segment [{a}, {b}]: trapesoid method value : {trapesoid_value}, real value : {real_value}".format(
#		func=func, a=a, b=b, trapesoid_value = trapesoid_integrate(func,a, b, 0.001), real_value = float(definite_integral(func, x, a, b))
#	)

func = x^4
a = 1.
b = 2.
points = []
for segments_count in xrange(2, 500):
	step = (b - a) / segments_count
	error = abs(float(definite_integral(func, x, a, b)) - trapesoid_integrate(func, a, b, step))
	points.append((log(segments_count), log(error)))
line(points)

