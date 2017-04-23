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

#
#class LogRegL2OptimizedOracle(LogRegL2Oracle):
#	"""
#	Oracle for logistic regression with l2 regularization
#	with optimized *_directional methods (are used in line_search).
#
#	For explanation see LogRegL2Oracle.
#	"""
#	def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=None):
#		super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
#		self.Abx = None
#		self.last_x = None
#		self.last_d = None
#		self.Abd = None
#		self.last_v = None
#		self.Abv = None
#		self.cur_Abx = None
#		self.last_x2 = None
#		self.last_v2 = None
#
#	def store_x(self, x):
#		if not np.array_equal(self.last_x, x) or not np.array_equal(self.last_v, x):
#			self.last_x = np.copy(x)
#			self.Abx = (-self.b) * self.matvec_Ax(x)
#		elif np.array_equal(self.last_v, x):
#			self.Abx, self.Abv = self.Abv, self.Abx
#			self.last_x, self.last_v = self.last_v, self.last_x
#
#		if np.array_equal(self.last_x, x) or np.array_equal(self.last_v, x):
#			print("optimized")
#
#	def store_d(self, d):
#		if not np.array_equal(self.last_d, d):
#			self.last_d = np.copy(d)
#			self.Abd = (-self.b) * self.matvec_Ax(d)
#
#	def func(self, x, store=True):
#		self.store_x(x)
#		return 1.0 / self.b.shape[0] * np.sum(np.logaddexp(np.zeros(self.b.shape[0]), self.Abx)) + self.regcoef * 0.5 * x.dot(x)
#
#	def grad(self, x, store=True):
#		self.store_x(x)
#		return (-1.0/self.b.shape[0]) * self.matvec_ATx(scipy.special.expit(self.Abx) * self.b) + self.regcoef * x
#
#	def hess(self, x):
#		self.store_x(x)
#		return (1.0/self.b.shape[0]) * self.matmat_ATsA(scipy.special.expit(-self.Abx) * scipy.special.expit(self.Abx)) + \
#			self.regcoef * np.identity(x.shape[0])
#
#	def func_directional(self, x, d, alpha):
#		self.store_x(x)
#		self.store_d(d)
#		v = self.Abx + alpha * self.Abd
#		x = x + alpha * d
#		self.last_v = np.copy(x)
#		self.Abv = np.copy(v)
#		return 1.0 / self.b.shape[0] * np.sum(np.logaddexp(np.zeros(self.b.shape[0]), v)) + self.regcoef * 0.5 * x.dot(x)
#
#
#	def grad_directional(self, x, d, alpha):
#		self.store_x(x)
#		self.store_d(d)
#		v = self.Abx + alpha * self.Abd
#		x = x + alpha * d
#
#		self.last_v = np.copy(x)
#		self.Abv = np.copy(v)
#
#		return ((-1.0/self.b.shape[0]) * self.matvec_ATx(scipy.special.expit(v) * self.b) + self.regcoef * x).dot(d)
#

class LogRegL2OptimizedOracle(LogRegL2Oracle):
	"""
	Oracle for logistic regression with l2 regularization
	with optimized *_directional methods (are used in line_search).
	For explanation see LogRegL2Oracle.
	"""
	def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
		self.matmat_diagbA = lambda x: self.diag_b.dot(x)
		self.diag_b = scipy.sparse.diags(b, 0)
		self.last_x = None
		self.last_d = None
		self.last_alpha = 0
		self.last_Ax = None
		self.last_Ad = None

#		self.memoized = {
#			'a' : 0,
#			'x' : None,
#			'd' : None,
#			'Ax': None,
#			'Ad': None
#		}
		super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

	def update_opt_x(self, x):
		if np.allclose(np.zeros(x.shape[0]), x):
			return np.zeros(self.b.shape[0])
		if self.last_x is None:
			self.last_x = np.copy(x)
			self.last_Ax = -self.b *self.matvec_Ax(x)
		if not np.allclose(self.last_x, x):
			if np.allclose(self.last_x + self.last_alpha * self.last_d, x):
				self.last_Ax = self.last_Ax + self.last_alpha * self.last_Ad
			else:
				self.last_Ax = -self.b * self.matvec_Ax(x)

			self.last_x = np.copy(x)

		return self.last_Ax

	def update_opt_d(self, d):
		if np.allclose(np.zeros(d.shape[0]), d):
			return np.zeros(self.b.shape[0])
		if self.last_d is None:
			self.last_d = np.copy(d) ###<<<<
			self.last_Ad = -self.b * self.matvec_Ax(d)
		if not np.allclose(self.last_d, d):
			self.last_d = np.copy(d)
			self.last_Ad = -self.b * self.matvec_Ax(d)

		return self.last_Ad

#	def o(self, entity, rvalue):
#		if np.allclose(np.zeros(rvalue.shape[0]), rvalue):
#			return np.zeros(self.b.shape[0])
#		if entity == 'Ad':
#			if self.memoized['d'] is None:
#				self.memoized['d'] = np.copy(rvalue)
#				self.memoized['Ad'] = self.matvec_Ax(rvalue)
#				return self.memoized['Ad']
#			if np.allclose(self.memoized['d'], rvalue):
#				print("opts")
#				return self.memoized['Ad']
#			else:
#				self.memoized['d'] = np.copy(rvalue)
#				self.memoized['Ad'] = self.matvec_Ax(rvalue)
#				return self.memoized['Ad']
#		elif entity == 'Ax':
#			if self.memoized['x'] is None:
#				self.memoized['x'] = np.copy(rvalue)
#				self.memoized['Ax'] = self.matvec_Ax(rvalue)
#				return self.memoized['Ax']
#			if np.allclose(self.memoized['x'], rvalue):
#				print("opts")
#				return self.memoized['Ax']
#			else:
#				if np.allclose(self.memoized['x'] + self.memoized['a']*self.memoized['d'], rvalue):
#					self.memoized['x'] = np.copy(rvalue)
#					self.memoized['Ax'] = self.memoized['Ax'] + self.memoized['a']*self.memoized['Ad']
#					print("opts")
#					return self.memoized['Ax']
#				else:
#					self.memoized['x'] = np.copy(rvalue)
#					self.memoized['Ax'] = self.matvec_Ax(rvalue)
#					return self.memoized['Ax']
#
#	def dpsi(self, x, d, alpha):
#		arg = self.o('Ax', x) + alpha * self.o('Ad', d)
#		m = np.alen(self.b)
#		x = np.array(x)
#		c = self.matmat_diagbA(
#			np.vectorize(scipy.special.expit)(self.matmat_diagbA(
#				arg
#			) * (-1)
#											  )
#		) * (-1 / m)
#		self.last_alpha = alpha
#		return c

	def func(self, x):
		return self.func_directional(x, np.zeros(x.shape[0]), 0)

#	def grad(self, x):
#		return self.matvec_ATx(self.dpsi(x, np.zeros(x.shape[0]), 0)) + x * self.regcoef


	def func_directional(self, x, d, alpha):
		Av = self.update_opt_x(x) + alpha * self.update_opt_d(d)
		v = x + d * alpha
		func_res = 1.0 / self.b.shape[0] * np.sum(np.logaddexp(np.zeros(self.b.shape[0]), Av)) + self.regcoef * 0.5 * v.dot(v)
		self.last_alpha = alpha
		return np.squeeze(func_res)

	def grad_directional(self, x, d, alpha):
#        m = np.alen(self.b)
#        n = np.alen(x)
#        c = self.matmat_ATsA(
#            np.vectorize(lambda x: scipy.special.expit(x)*(1-scipy.special.expit(x)))(
#                self.matmat_diagbA(
#                    self.matvec_Ax(x)
#                )*(-1)
#            )
#        )*(1/m) + np.eye(n) * self.regcoef
#        return c


		Av = self.update_opt_x(x) + alpha * self.update_opt_d(d)
		v = x + d * alpha
		return ((-1.0/self.b.shape[0]) * self.matvec_ATx(scipy.special.expit(Av) * self.b) + self.regcoef * v).dot(d)

#		Av = self.update_opt_x(x) + alpha * self.update_opt_d(d)
#		return np.squeeze(self.grad(x + alpha * d).dot(self.update_opt_d(d))
#		return np.squeeze(self.dpsi(x, d, alpha).dot(self.o('Ad', d)) + (x + d * alpha).dot(d) * self.regcoef)

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

######################################################################
#import numpy as np
#import scipy
#import scipy.sparse as sp
#from scipy.special import expit
#
#
#class BaseSmoothOracle(object):
#    """
#    Base class for implementation of oracles.
#    """
#    def func(self, x):
#        """
#        Computes the value of function at point x.
#        """
#        raise NotImplementedError('Func oracle is not implemented.')
#
#    def grad(self, x):
#        """
#        Computes the gradient at point x.
#        """
#        raise NotImplementedError('Grad oracle is not implemented.')
#
#    def hess(self, x):
#        """
#        Computes the Hessian matrix at point x.
#        """
#        raise NotImplementedError('Hessian oracle is not implemented.')
#
#    def func_directional(self, x, d, alpha):
#        """
#        Computes phi(alpha) = f(x + alpha*d).
#        """
#        return np.squeeze(self.func(x + alpha * d))
#
#    def grad_directional(self, x, d, alpha):
#        """
#        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
#        """
#        return np.squeeze(self.grad(x + alpha * d).dot(d))
#
#
#class QuadraticOracle(BaseSmoothOracle):
#    """
#    Oracle for quadratic function:
#       func(x) = 1/2 x^TAx - b^Tx.
#    """
#
#    def __init__(self, A, b):
#        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
#            raise ValueError('A should be a symmetric matrix.')
#        self.A = A
#        self.b = b
#
#    def func(self, x):
#        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)
#
#    def grad(self, x):
#        return self.A.dot(x) - self.b
#
#    def hess(self, x):
#        return self.A
#
#
#class LogRegL2Oracle(BaseSmoothOracle):
#    """
#    Oracle for logistic regression with l2 regularization:
#         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.
#    Let A and b be parameters of the logistic regression (feature matrix
#    and labels vector respectively).
#    For user-friendly interface use create_log_reg_oracle()
#    Parameters
#    ----------
#        matvec_Ax : function
#            Computes matrix-vector product Ax, where x is a vector of size n.
#        matvec_ATy : function of y
#            Computes matrix-vector product A^Ty, where y is a vector of size m.
#        matmat_ATsA : function
#            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
#    """
#    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
#        self.matvec_Ax = matvec_Ax
#        self.matvec_ATx = matvec_ATx
#        self.matmat_ATsA = matmat_ATsA
#        self.diag_b = sp.diags(b, 0)
#        self.matmat_diagbA = lambda x: self.diag_b.dot(x)
#        self.b = b
#        self.regcoef = regcoef
#        if regcoef is None:
#            self.regcoef = 1 / self.b.shape[0]
#        else:
#            self.regcoef = regcoef
#
#
#    def func(self, x):
#        c = np.mean(np.vectorize(lambda x: np.logaddexp(0, x))(self.matmat_diagbA(self.matvec_Ax(x)) * (-1))) + np.dot(x, x) * self.regcoef / 2
#        return c
#
#    def grad(self, x):
#        m = np.alen(self.b)
#        x = np.array(x)
#        c = self.matvec_ATx(
#            self.matmat_diagbA(
#                np.vectorize(scipy.special.expit)( self.matmat_diagbA(
#                        self.matvec_Ax(x)
#                    ) * (-1)
#                )
#            )
#        )*(-1/m) + x * self.regcoef
#        # -1/m * AT*Diag(b)*sigma(-Diag(b)*A*x) + lambda*x
#        return c
#
#    def hess(self, x):
#        m = np.alen(self.b)
#        n = np.alen(x)
#        c = self.matmat_ATsA(
#            np.vectorize(lambda x: scipy.special.expit(x)*(1-scipy.special.expit(x)))(
#                self.matmat_diagbA(
#                    self.matvec_Ax(x)
#                )*(-1)
#            )
#        )*(1/m) + np.eye(n) * self.regcoef
#        return c
#
#
#class LogRegL2OptimizedOracle(LogRegL2Oracle):
#    """
#    Oracle for logistic regression with l2 regularization
#    with optimized *_directional methods (are used in line_search).
#    For explanation see LogRegL2Oracle.
#    """
#    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
#        self.memoized = {
#            'a' : 0,
#            'x' : None,
#            'd' : None,
#            'Ax': None,
#            'Ad': None
#        }
#        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
#
#    def o(self, entity, rvalue):
#        if np.allclose(np.zeros(rvalue.shape[0]), rvalue):
#            return np.zeros(self.b.shape[0])
#        if entity == 'Ad':
#            if self.memoized['d'] is None:
#                self.memoized['d'] = np.copy(rvalue)
#                self.memoized['Ad'] = self.matvec_Ax(rvalue)
#                return self.memoized['Ad']
#            if np.allclose(self.memoized['d'], rvalue):
#                return self.memoized['Ad']
#            else:
#                self.memoized['d'] = np.copy(rvalue)
#                self.memoized['Ad'] = self.matvec_Ax(rvalue)
#                return self.memoized['Ad']
#        elif entity == 'Ax':
#            if self.memoized['x'] is None:
#                self.memoized['x'] = np.copy(rvalue)
#                self.memoized['Ax'] = self.matvec_Ax(rvalue)
#                return self.memoized['Ax']
#            if np.allclose(self.memoized['x'], rvalue):
#                return self.memoized['Ax']
#            else:
#                if np.allclose(self.memoized['x'] + self.memoized['a']*self.memoized['d'], rvalue):
#                    self.memoized['x'] = np.copy(rvalue)
#                    self.memoized['Ax'] = self.memoized['Ax'] + self.memoized['a']*self.memoized['Ad']
#                    return self.memoized['Ax']
#                else:
#                    self.memoized['x'] = np.copy(rvalue)
#                    self.memoized['Ax'] = self.matvec_Ax(rvalue)
#                    return self.memoized['Ax']
#
#    def dpsi(self, x, d, alpha):
#        arg = self.o('Ax', x) + alpha * self.o('Ad', d)
#        m = np.alen(self.b)
#        x = np.array(x)
#        c = self.matmat_diagbA(
#            np.vectorize(scipy.special.expit)(self.matmat_diagbA(
#                arg
#            ) * (-1)
#                                              )
#        ) * (-1 / m)
#        self.memoized['a'] = alpha
#        return c
#
#    def func(self, x):
#        return self.func_directional(x, np.zeros(x.shape[0]), 0)
#
#    def grad(self, x):
#        return self.matvec_ATx(self.dpsi(x, np.zeros(x.shape[0]), 0)) + x * self.regcoef
#
#    def hess(self, x):
#        m = np.alen(self.b)
#        n = np.alen(x)
#        c = self.matmat_ATsA(
#            np.vectorize(lambda x: scipy.special.expit(x) * (1 - scipy.special.expit(x)))(
#                self.matmat_diagbA(
#                    self.o('Ax', x)
#                ) * (-1)
#            )
#        ) * (1 / m) + np.eye(n) * self.regcoef
#        return c
#
#    def func_directional(self, x, d, alpha):
#        arg = self.o('Ax', x) + alpha * self.o('Ad', d)
#        c = np.mean(np.vectorize(lambda x: np.logaddexp(0, x))(self.matmat_diagbA(arg) * (-1))) + np.dot(x + d * alpha, x + d * alpha) * self.regcoef / 2
#        self.memoized['a'] = alpha
#        return np.squeeze(c)
#
#    def grad_directional(self, x, d, alpha):
#        return np.squeeze(self.dpsi(x, d, alpha).dot(self.o('Ad', d)) + (x + d * alpha).dot(d) * self.regcoef)
#
#
#def create_log_reg_oracle(A, b, regcoef=None, oracle_type='usual'):
#    """
#    Auxiliary function for creating logistic regression oracles.
#        `oracle_type` must be either 'usual' or 'optimized'
#    """
#    matvec_Ax = lambda x: A.dot(x)
#    matvec_ATx = lambda x: A.T.dot(x)
#
#    def matmat_ATsA(s):
#        return A.T.dot(sp.diags(s, 0).dot(A))
#
#    if oracle_type == 'usual':
#        oracle = LogRegL2Oracle
#    elif oracle_type == 'optimized':
#        oracle = LogRegL2OptimizedOracle
#    else:
#        raise 'Unknown oracle_type=%s' % oracle_type
#    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
#
#
#
#def grad_finite_diff(func, x, eps=1e-8):
#    """
#    Returns approximation of the gradient using finite differences:
#        result_i := (f(x + eps * e_i) - f(x)) / eps,
#        where e_i are coordinate vectors:
#        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
#                          >> i <<
#    """
#    n = np.alen(x)
#    c = np.apply_along_axis(lambda z: (func(x + z*eps)-func(x)) / eps, 1, np.eye(n))
#    return c
#
#
#def hess_finite_diff(func, x, eps=1e-5):
#    """
#    Returns approximation of the Hessian using finite differences:
#        result_{ij} := (f(x + eps * e_i + eps * e_j)
#                               - f(x + eps * e_i)
#                               - f(x + eps * e_j)
#                               + f(x)) / eps^2,
#        where e_i are coordinate vectors:
#        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
#                          >> i <<
#    """
#    n = np.alen(x)
#    E = np.eye(n)
#    hess = np.array([[ (func(x + e_i*eps + e_j*eps) - func(x + e_i*eps) - func(x + e_j*eps) + func(x))/eps**2 for e_i in E ] for e_j in E])
#    return hess
