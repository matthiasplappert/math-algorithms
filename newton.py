import timeit

import numpy as np
from scipy.linalg import lu_factor, lu_solve


def check_jacobian(x0, func, jacobian, eps=1e-5):
	if x0.ndim != 1:
		raise ValueError('x0 must be a vector')

	# Compute analytic gradients
	J = jacobian(x0)
	n_out, n_in = J.shape
	if x0.shape[0] != n_in:
		raise ValueError('x0 must match Jacobian')

	# Compute approximate Jacobian
	approx_J = np.zeros(J.shape)
	for col_idx in xrange(n_in):
		left = np.copy(x0)
		right = np.copy(x0)
		left[col_idx] -= eps
		right[col_idx] += eps
		approx = (func(right) - func(left)) / (2. * eps)
		approx_J[:, col_idx] = approx
	return np.allclose(J, approx_J)


def newton(x0, func, jacobian, tol=1e-2, verbose=False):
	dx = None
	x = np.copy(x0)
	step = 0
	while dx is None or np.linalg.norm(dx) > tol:
		step += 1
		dx = np.linalg.solve(jacobian(x), -func(x))
		x += dx
		if verbose:
			print('step %.3d: %s' % (step, x))
	return x


def simplified_newton(x0, func, jacobian, tol=1e-2, verbose=False):
	dx = None
	x = np.copy(x0)
	step = 0
	factor = lu_factor(jacobian(x0))
	while dx is None or np.linalg.norm(dx) > tol:
		step += 1
		dx = lu_solve(factor, -func(x))
		x += dx
		if verbose:
			print('step %.3d: %s' % (step, x))
	return x


def main():
	func = lambda x: np.array([np.sqrt(x[0]) - np.sin(x[1]), x[0] ** 2. + x[1] ** 2. - 1])
	jacobian = lambda x: np.array([[1. / (2. * np.sqrt(x[0])), -np.cos(x[1])],
								   [2. * x[0], 				   2. * x[1]]])
	x0 = np.array([.5, np.pi / 2.])

	if not check_jacobian(np.random.random(2), func, jacobian):
		exit('Jacobian seems to be wrong')

	print('solving using Newton ...')
	start = timeit.default_timer()
	x = newton(x0, func, jacobian, tol=1e-10, verbose=True)
	if np.allclose(func(x), 0.):
		print('solution: %s' % x)
	else:
		print('failed')
	print('done, took %fs' % (timeit.default_timer() - start))
	print('')

	print('solving using simplified Newton ...')
	start = timeit.default_timer()
	x = simplified_newton(x0, func, jacobian, tol=1e-10, verbose=True)
	if np.allclose(func(x), 0.):
		print('solution: %s' % x)
	else:
		print('failed')
	print('done, took %fs' % (timeit.default_timer() - start))
	print('')


if __name__ == '__main__':
	main()
