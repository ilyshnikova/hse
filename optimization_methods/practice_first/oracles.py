import numpy as np
import scipy
from scipy.special import expit
import scipy as sp


class BaseSmoothOracle(object):
	"""
	Base class for implementation of oracles.
	"""
	def func(self, x):
		"""
		Computes the value of function at point x.
		"""
		raise NotImplementedError('Func oracle is not implemented.')

	def grad(self, x):
		"""
		Computes the gradient at point x.
		"""
		raise NotImplementedError('Grad oracle is not implemented.')

	def hess(self, x):
		"""
		Computes the Hessian matrix at point x.
		"""
		raise NotImplementedError('Hessian oracle is not implemented.')

	def func_directional(self, x, d, alpha):
		"""
		Computes phi(alpha) = f(x + alpha*d).
		"""
		return np.squeeze(self.func(x + alpha * d))

	def grad_directional(self, x, d, alpha):
		"""
		Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
		"""
		return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
	"""
	Oracle for quadratic function:
	   func(x) = 1/2 x^TAx - b^Tx.
	"""

	def __init__(self, A, b):
		if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
			raise ValueError('A should be a symmetric matrix.')
		self.A = A
		self.b = b

	def func(self, x):
		return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

	def grad(self, x):
		return self.A.dot(x) - self.b

	def hess(self, x):
		return self.A


class LogRegL2Oracle(BaseSmoothOracle):
	"""
	Oracle for logistic regression with l2 regularization:
		 func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

	Let A and b be parameters of the logistic regression (feature matrix
	and labels vector respectively).
	For user-friendly interface use create_log_reg_oracle()

	Parameters
	----------
		matvec_Ax : function
			Computes matrix-vector product Ax, where x is a vector of size n.
		matvec_ATx : function of x
			Computes matrix-vector product A^Tx, where x is a vector of size m.
		matmat_ATsA : function
			Computes matrix-matrix-matrix product A^T * Diag(s) * A,
	"""
	def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=None):
		self.matvec_Ax = matvec_Ax
		self.matvec_ATx = matvec_ATx
		self.matmat_ATsA = matmat_ATsA
		self.b = b
		if regcoef is None:
			self.regcoef = 1 / self.b.shape[0]
		else:
			self.regcoef = regcoef

	def func(self, x):
		return 1.0 / self.b.shape[0] * np.sum(np.logaddexp(np.zeros(self.b.shape[0]), -self.b * self.matvec_Ax(x))) + self.regcoef * 0.5 * x.dot(x)

	def grad(self, x):
		return (-1.0/self.b.shape[0]) * self.matvec_ATx(scipy.special.expit(-1 * self.b * self.matvec_Ax(x)) * self.b) + self.regcoef * x

	def hess(self, x):
		Ax = self.matvec_Ax(x)
		return (1.0/self.b.shape[0]) * self.matmat_ATsA(scipy.special.expit(Ax * self.b) * scipy.special.expit(-Ax * self.b)) + \
			self.regcoef * np.identity(x.shape[0])

class LogRegL2OptimizedOracle(LogRegL2Oracle):
	"""
	Oracle for logistic regression with l2 regularization
	with optimized *_directional methods (are used in line_search).
	For explanation see LogRegL2Oracle.
	"""
	def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
		self.last_x = None
		self.last_d = None
		self.last_alpha = 0
		self.last_Ax = None
		self.last_Ad = None

		super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

	def update_opt_x(self, x):
		if self.last_x is None or not np.allclose(self.last_x, x):
			self.last_x = np.copy(x)
			self.last_Ax = self.matvec_Ax(x)
		elif self.last_d is not None and np.allclose(self.last_x + self.last_alpha * self.last_d, x):
			self.last_Ax = self.last_Ax + self.last_alpha * self.last_Ad
			self.last_x = np.copy(x)

		return self.last_Ax

	def update_opt_d(self, d):
		if self.last_x is not None and np.allclose(self.last_x, d):
			self.last_d = np.copy(d)
			self.last_Ad = self.last_Ax
		if self.last_d is None or not np.allclose(self.last_d, d):
			self.last_d = np.copy(d)
			self.last_Ad = self.matvec_Ax(d)
		return self.last_Ad

	def func(self, x):
		Ax = self.update_opt_x(x)
		return 1.0 / self.b.shape[0] * np.sum(np.logaddexp(np.zeros(self.b.shape[0]), -self.b * Ax)) + self.regcoef * 0.5 * x.dot(x)

	def grad(self, x):
		Ax = self.update_opt_x(x)
		return (-1.0/self.b.shape[0]) * self.matvec_ATx(scipy.special.expit(-1 * self.b * Ax) * self.b) + self.regcoef * x

	def hess(self, x):
		Ax = self.update_opt_x(x)
		return (1.0/self.b.shape[0]) * self.matmat_ATsA(scipy.special.expit(Ax * self.b) * scipy.special.expit(-Ax * self.b)) + \
			self.regcoef * np.identity(x.shape[0])

	def func_directional(self, x, d, alpha):
		Av = self.b * (self.update_opt_x(x) + alpha * self.update_opt_d(d))
		v = x + d * alpha
		func_res = 1.0 / self.b.shape[0] * np.sum(np.logaddexp(np.zeros(self.b.shape[0]), -Av)) + self.regcoef * 0.5 * v.dot(v)
		self.last_alpha = alpha
		return np.squeeze(func_res)

	def grad_directional(self, x, d, alpha):
		Av = self.b * (self.update_opt_x(x) + alpha * self.update_opt_d(d))
		v = x + d * alpha

#		return -(self.b * (scipy.special.expit(-self.b * next_ax))).dot(
#		    self.matvec_Ax(d)
#		    ) / self.b.size + self.regcoef * next_x.T.dot(d)

		return  -(self.b * (scipy.special.expit(-Av))).dot(self.last_Ad) / self.b.size + self.regcoef * v.T.dot(d)
#		return (-1.0 / self.b.shape[0]) * ((self.b * scipy.special.expit(-1 * self.b * Av))).dot(self.last_Ad) + self.regcoef * v.dot(d)
#		return (1.0/self.b.shape[0]) * (scipy.special.expit(Av) * self.b) * self.matvec_Ax(d) + (self.regcoef * v).dot(d)
#		return ((-1.0/self.b.shape[0]) * self.matvec_ATx(scipy.special.expit(Av) * self.b) + self.regcoef * v).dot(d)

def create_log_reg_oracle(A, b, regcoef=None, oracle_type='usual'):
	"""
	Auxiliary function for creating logistic regression oracles.
		`oracle_type` must be either 'usual' or 'optimized'
	"""
	matvec_Ax = lambda x:  A.dot(x)
	matvec_ATx = lambda x: A.T.dot(x)
	def matmat_ATsA(s):
		if isinstance(A, sp.sparse.csr_matrix):
			return A.T.dot(scipy.sparse.diags(s)).dot(A)
		return A.T.dot(np.diag(s)).dot(A)

	if oracle_type == 'usual':
		oracle = LogRegL2Oracle
	elif oracle_type == 'optimized':
		oracle = LogRegL2OptimizedOracle
	else:
		raise 'Unknown oracle_type=%s' % oracle_type
	return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)



def grad_finite_diff(func, x, eps=1e-8):
	"""
	Returns approximation of the gradient using finite differences:
		result_i := (f(x + eps * e_i) - f(x)) / eps,
		where e_i are coordinate vectors:
		e_i = (0, 0, ..., 0, 1, 0, ..., 0)
						  >> i <<
	"""
	e = np.identity(x.size)
	return np.array([(func(x + eps * e[i]) - func(x)) / eps for i in range(x.size)])


def hess_finite_diff(func, x, eps=1e-5):
	"""
	Returns approximation of the Hessian using finite differences:
		result_{ij} := (f(x + eps * e_i + eps * e_j)
							   - f(x + eps * e_i)
							   - f(x + eps * e_j)
							   + f(x)) / eps^2,
		where e_i are coordinate vectors:
		e_i = (0, 0, ..., 0, 1, 0, ..., 0)
						  >> i <<
	"""
	e = np.identity(x.size)
	return np.array([[(func(x + eps * e[i] + eps * e[j]) - func(x + eps * e[i]) - func(x + eps * e[j]) + func(x)) / eps**2 for i in range(x.size)] for j in range(x.size)])
