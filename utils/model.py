
from collections.abc import Callable

import numba
import numpy as np


def cosine_kernel(A: float, N: int) -> np.ndarray:
    """
    Calculate the distance kernel delta_y for a system of oscillators.
    This function generates a coupling kernel based on a cosine function,
    which is then normalized.

    :param A: A scaling parameter that affects the coupling strength between oscillators.
    :param N: The number of spatial grid points, which corresponds to the number of oscillators.
    :returns: A 1D numpy array of length N containing the computed coupling kernel values, scaled to ensure normalization.
    """
    x = np.linspace(-np.pi, np.pi, N, endpoint=False)
    delta_y = (1 + A * np.cos(x)) / (2 * np.pi)
    kernel = delta_y * ((2 * np.pi) / N)
    return kernel


def cosine_kernel_fft(A: float, N: int) -> np.ndarray:
    """
    Calculate the FFT of the shifted cosine kernel for a system of oscillators.
    This function generates a coupling kernel based on a cosine function,
    shifts it for convolution purposes, and computes its FFT.

    :param A: A scaling parameter that affects the coupling strength between oscillators.
    :param N: The number of spatial grid points, which corresponds to the number of oscillators.
    :returns: A 1D numpy array of length N containing the FFT of the shifted coupling kernel.
    """
    kernel = cosine_kernel(A, N)
    # Shift the kernel for convolution theorem
    kernel_shifted = np.fft.ifftshift(kernel)
    kernel_fft = np.fft.fft(kernel_shifted)
    return kernel_fft


def kuramoto_kernel(
    omegas: np.ndarray | float,
    kernel_fft: np.ndarray,
    alpha: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Kuramoto model using FFT-based convolution.

    :param omega: Natural frequencies array (length N)
    :param alpha: Phase lag parameter
    :param kernel_fft: Precomputed FFT of the shifted coupling kernel
    """

    def f(theta: np.ndarray) -> np.ndarray:
        u = np.exp(1j * theta)  # e^{iθ}

        # Z = G * e^{iθ} via convolution theorem
        Z = np.fft.ifft(np.fft.fft(u) * kernel_fft)

        # Kuramoto coupling: Im[Z * e^{-i(θ + α)}]
        coupling = np.imag(Z * np.exp(-1j * (theta + alpha)))

        return omegas + coupling

    return f


def vf_r_theta(
    omega: np.ndarray, Omega: float, R: np.ndarray, Theta: np.ndarray, alpha: float
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Vector field for phase dynamics using precomputed order parameter.

    :param omega: Natural frequencies array (length N)
    :param Omega: Collective frequency 
    :param R: Order parameter amplitude array (length N)
    :param Theta: Order parameter phase array (length N)
    :param alpha: Phase lag parameter
    """

    @numba.jit(nopython=True)
    def equation(theta: np.ndarray) -> np.ndarray:
        return omega - Omega + R * np.sin(Theta - theta - alpha)  # type:ignore

    return equation  # type:ignore
