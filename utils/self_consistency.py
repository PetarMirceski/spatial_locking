import cmath
from collections.abc import Callable

import numpy as np
from numba import njit
from scipy.optimize import OptimizeResult, root


@njit
def expression(theta: np.ndarray, delta: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Compute the complex expression for self-consistency iteration.

    :param theta: Phase array (length N)
    :param delta: Frequency detuning array (length N)
    :param r: Order parameter amplitude array (length N)
    :returns: Complex expression array (length N)
    """
    result = np.empty_like(delta, dtype=np.complex128)
    for i in range(delta.size):
        sqrt_term = cmath.sqrt(delta[i] ** 2 - r[i] ** 2)
        result[i] = np.exp(1j * theta[i]) * (delta[i] - sqrt_term) / r[i]
    return result


def right_equation_fft(
    omega: float,
    r: np.ndarray,
    theta: np.ndarray,
    kernel_fft: np.ndarray,
    alpha: float,
    nat_freqs: np.ndarray,
) -> np.ndarray:
    """
    Compute the right-hand side of the self-consistency equation via FFT.

    :param omega: Collective frequency
    :param r: Order parameter amplitude array (length N)
    :param theta: Order parameter phase array (length N)
    :param kernel_fft: Precomputed FFT of the coupling kernel
    :param alpha: Phase lag parameter
    :param nat_freqs: Natural frequencies array (length N)
    :returns: Right-hand side complex array (length N)
    """
    beta = (np.pi / 2) - alpha
    delta = nat_freqs - omega

    expression_value = expression(theta, delta, r)

    # FFT-based convolution: G * expression_value
    right_side = np.fft.ifft(np.fft.fft(expression_value) * kernel_fft)

    right_side *= np.exp(1j * beta)
    return right_side


def _pack(r: np.ndarray, theta: np.ndarray, omega: float, pin: int) -> np.ndarray:
    return np.concatenate([r, np.delete(theta, pin), [omega]])


def _unpack(
    x: np.ndarray, n: int, pin: int, theta_pin: float
) -> tuple[np.ndarray, np.ndarray, float]:
    r = x[:n]
    theta = np.insert(x[n : 2 * n - 1], pin, theta_pin)
    return r, theta, x[-1]


def _solve_root(
    rhs: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    r_init: np.ndarray,
    theta_init: np.ndarray,
    omega_init: float,
    options: dict,
) -> tuple[np.ndarray, np.ndarray, float, OptimizeResult]:
    """
    Solve R e^{i Theta} = rhs(Omega, R, Theta) for R, Theta and Omega with
    scipy's Jacobian-free Newton-Krylov method.

    The solver stops once ALL four criteria hold (max-norm):
    |F| <= fatol, |F| <= ftol*|F0|, |dx| <= xatol, |dx| <= xtol*|x|.
    Pass np.inf to switch a criterion off.

    The global phase symmetry Theta -> Theta + c is removed by pinning Theta at
    argmin(theta_init) to its initial value, which leaves 2N real unknowns
    (N for R, N-1 for Theta, 1 for Omega) for the 2N real equations.
    """
    n = r_init.shape[0]
    pin = int(np.argmin(theta_init))
    phase_pin = np.exp(-1j * theta_init[pin])

    def residual(u):
        z = u[:n] + 1j * u[n : 2 * n]
        omega = u[-1]
        f = z - rhs(omega, np.abs(z), np.angle(z))
        # 2N equations + 1 phase pin, for 2N+1 unknowns
        return np.concatenate([f.real, f.imag, [(z[pin] * phase_pin).imag]])

    z0 = r_init * np.exp(1j * theta_init)
    u0 = np.concatenate([z0.real, z0.imag, [omega_init]])
    sol = root(residual, u0, method="krylov", options=options)

    z = sol.x[:n] + 1j * sol.x[n : 2 * n]
    theta = np.unwrap(np.angle(z))
    return np.abs(z), theta - theta.min(), float(sol.x[-1]), sol


def self_consistency_root(
    r_init: np.ndarray,
    theta_init: np.ndarray,
    omega_init: float,
    kernel_fft: np.ndarray,
    nat_freqs: np.ndarray,
    alpha: float,
    options: dict,
) -> tuple[np.ndarray, np.ndarray, float, OptimizeResult]:
    """
    Solve the self-consistency equations directly with Newton-Krylov.

    Solves  R(x) e^{i Theta(x)} = e^{i beta} G * [e^{i Theta} (Delta - sqrt(Delta^2 - R^2)) / R]
    for R(x), Theta(x) and Omega simultaneously, instead of bisecting on Omega.

    :param r_init: Initial guess for R(x) (length N)
    :param theta_init: Initial guess for Theta(x) (length N)
    :param omega_init: Initial guess for the collective frequency Omega
    :param kernel_fft: Precomputed FFT of the coupling kernel
    :param nat_freqs: Natural frequencies array (length N)
    :param alpha: Phase lag parameter
    :param fatol: Absolute residual tolerance, max|F(x)| <= fatol
    :param ftol: Relative residual tolerance, max|F(x)| <= ftol * max|F(x0)|
    :param xatol: Absolute step tolerance, max|dx| <= xatol
    :param xtol: Relative step tolerance, max|dx| <= xtol * max|x|
    :param maxiter: Maximum Newton iterations (None: scipy's default)
    :returns: Tuple of (R, Theta - min(Theta), Omega, scipy OptimizeResult)
    """

    def rhs(omega: float, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return right_equation_fft(omega, r, theta, kernel_fft, alpha, nat_freqs)

    return _solve_root(rhs, r_init, theta_init, omega_init, options)


@njit
def expression_rewrite(theta: np.ndarray, delta: float, r: np.ndarray) -> np.ndarray:
    """
    Compute the complex expression for self-consistency with uniform frequency.

    :param theta: Phase array (length N)
    :param delta: Scalar frequency detuning
    :param r: Order parameter amplitude array (length N)
    :returns: Complex expression array (length N)
    """
    result = np.empty_like(r, dtype=np.complex128)
    for i in range(r.size):
        sqrt_term = cmath.sqrt(delta**2 - r[i] ** 2)
        result[i] = np.exp(1j * theta[i]) * ((delta - sqrt_term) / r[i])
    return result


def right_equation_rewrite_fft(
    omega: float,
    r: np.ndarray,
    theta: np.ndarray,
    kernel_fft: np.ndarray,
    alpha: float,
    nat_freq: float,
    epsh_func: np.ndarray,
) -> np.ndarray:
    """
    Compute the right-hand side of the perturbed self-consistency equation via FFT.

    :param omega: Collective frequency
    :param r: Order parameter amplitude array (length N)
    :param theta: Order parameter phase array (length N)
    :param kernel_fft: Precomputed FFT of the coupling kernel
    :param alpha: Phase lag parameter
    :param nat_freq: Uniform natural frequency
    :param epsh_func: Spatial forcing function epsilon * h(x) (length N)
    :returns: Right-hand side complex array (length N)
    """
    beta = (np.pi / 2) - alpha
    delta = nat_freq - omega

    expression_value = expression_rewrite(theta, delta, r)

    # FFT-based convolution: G * expression_value
    right_side = np.fft.ifft(np.fft.fft(expression_value) * kernel_fft)

    perturbation_term = epsh_func * expression_value
    right_side = np.exp(1j * beta) * right_side - perturbation_term
    return right_side


def self_consistency_rewrite_root(
    r_init: np.ndarray,
    theta_init: np.ndarray,
    omega_init: float,
    kernel_fft: np.ndarray,
    nat_freq: float,
    epsh_func: np.ndarray,
    alpha: float,
    options: dict,
) -> tuple[np.ndarray, np.ndarray, float, OptimizeResult]:
    """
    Root-finder version of self_consistency_rewrite_full (spatial forcing).

    Solves  R e^{i Theta} = e^{i beta} G * E - epsilon h(x) E,
    with E = e^{i Theta} (Delta - sqrt(Delta^2 - R^2)) / R, for R(x), Theta(x)
    and Omega simultaneously.

    :param r_init: Initial guess for R(x) (length N)
    :param theta_init: Initial guess for Theta(x) (length N)
    :param omega_init: Initial guess for the collective frequency Omega
    :param kernel_fft: Precomputed FFT of the coupling kernel
    :param nat_freq: Uniform natural frequency
    :param epsh_func: Spatial forcing function epsilon * h(x) (length N)
    :param alpha: Phase lag parameter
    :param fatol: Absolute residual tolerance, max|F(x)| <= fatol
    :param ftol: Relative residual tolerance, max|F(x)| <= ftol * max|F(x0)|
    :param xatol: Absolute step tolerance, max|dx| <= xatol
    :param xtol: Relative step tolerance, max|dx| <= xtol * max|x|
    :param maxiter: Maximum Newton iterations (None: scipy's default)
    :returns: Tuple of (R, Theta - min(Theta), Omega, scipy OptimizeResult)
    """

    def rhs(omega: float, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return right_equation_rewrite_fft(
            omega, r, theta, kernel_fft, alpha, nat_freq, epsh_func
        )

    return _solve_root(rhs, r_init, theta_init, omega_init, options)
