import cmath
from typing import List, Tuple

import numpy as np
from numba import njit


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


def self_consistency(
    r_orig: np.ndarray,
    theta_orig: np.ndarray,
    kernel_fft: np.ndarray,
    nat_freqs: np.ndarray,
    alpha: float,
    min_omega: float = 0.1,
    max_omega: float = 0.4,
    fit_itterations: int = 150,
    max_itterations: int = 60,
    smoothing_alpha: float = 0.6,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float]:
    """
    Fixed-point iteration to solve the self-consistency equations for chimera states.

    Uses binary search over omega combined with iterative smoothing to find
    the steady-state order parameter R(x), Theta(x) and collective frequency.

    :param r_orig: Initial order parameter amplitude array (length N)
    :param theta_orig: Initial order parameter phase array (length N)
    :param kernel_fft: Precomputed FFT of the coupling kernel
    :param nat_freqs: Natural frequencies array (length N)
    :param alpha: Phase lag parameter
    :param min_omega: Lower bound for binary search on omega
    :param max_omega: Upper bound for binary search on omega
    :param fit_itterations: Number of fixed-point iterations per omega value
    :param max_itterations: Number of binary search iterations
    :param smoothing_alpha: Smoothing factor for iteration updates (0 < alpha < 1)
    :returns: Tuple of (iteration history as list of (R, Theta) pairs, final omega)
    """
    omega = (min_omega + max_omega) / 2

    r0 = r_orig.copy()
    theta0 = theta_orig.copy()
    xp = []

    for itter in range(max_itterations):
        r0 = r_orig.copy()
        theta0 = theta_orig.copy()
        xp = []
        for _ in range(fit_itterations):
            z = smoothing_alpha * right_equation_fft(
                omega, r0, theta0, kernel_fft, alpha, nat_freqs
            ) + (1 - smoothing_alpha) * (r0 * np.exp(1j * theta0))
            r = np.abs(z)
            theta = np.angle(z)
            theta = theta - np.min(theta)
            r0 = r
            theta0 = theta
            xp.append((r0, theta0))

        if np.sum(xp[-1][0] - r_orig) / r_orig.shape[0] > 0:
            max_omega = omega
        else:
            min_omega = omega
        omega = (max_omega + min_omega) / 2
        print(f"itteration {itter}/{max_itterations}, omega={omega}")

    return xp, omega


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


def self_consistency_rewrite_full(
    r_orig: np.ndarray,
    theta_orig: np.ndarray,
    kernel_fft: np.ndarray,
    nat_freq: float,
    epsh_func: np.ndarray,
    alpha: float,
    min_omega: float = 0.1,
    max_omega: float = 0.4,
    fit_itterations: int = 150,
    max_itterations: int = 60,
    smoothing_alpha: float = 0.8,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], float]:
    """
    Fixed-point iteration for perturbed self-consistency with spatial forcing.

    Uses binary search over omega combined with iterative smoothing to find
    the steady-state order parameter R(x), Theta(x) and collective frequency
    in the presence of spatial forcing epsilon * h(x).

    :param r_orig: Initial order parameter amplitude array (length N)
    :param theta_orig: Initial order parameter phase array (length N)
    :param kernel_fft: Precomputed FFT of the coupling kernel
    :param nat_freq: Uniform natural frequency
    :param epsh_func: Spatial forcing function epsilon * h(x) (length N)
    :param alpha: Phase lag parameter
    :param min_omega: Lower bound for binary search on omega
    :param max_omega: Upper bound for binary search on omega
    :param fit_itterations: Number of fixed-point iterations per omega value
    :param max_itterations: Number of binary search iterations
    :param smoothing_alpha: Smoothing factor for iteration updates (0 < alpha < 1)
    :returns: Tuple of (iteration history as list of (R, Theta) pairs, final omega)
    """
    omega = (min_omega + max_omega) / 2

    r0 = r_orig.copy()
    theta0 = theta_orig.copy()
    xp = []

    for itter in range(max_itterations):
        r0 = r_orig.copy()
        theta0 = theta_orig.copy()
        xp = []
        for _ in range(fit_itterations):
            z = smoothing_alpha * right_equation_rewrite_fft(
                omega, r0, theta0, kernel_fft, alpha, nat_freq, epsh_func
            ) + (1 - smoothing_alpha) * (r0 * np.exp(1j * theta0))

            r = np.abs(z)
            theta = np.angle(z)
            theta = theta - np.min(theta)
            r0 = r
            theta0 = theta
            xp.append((r0, theta0))

        if np.sum(xp[-1][0] - r_orig) / r_orig.shape[0] > 0:
            max_omega = omega
        else:
            min_omega = omega
        omega = (max_omega + min_omega) / 2
        print(f"itteration {itter}/{max_itterations}, omega={omega}")

    return xp, omega
