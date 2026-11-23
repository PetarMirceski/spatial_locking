import numpy as np


def complex_param(phases: np.ndarray, kernel_fft: np.ndarray) -> np.ndarray:
    """
    Compute the complex order parameter Z via FFT-based convolution.

    :param phases: Phase array (length N)
    :param kernel_fft: Precomputed FFT of the coupling kernel
    :returns: Complex order parameter Z(x) array (length N)
    """
    u = np.exp(1j * phases)  # e^{iθ}
    # Z = G * e^{iθ}
    z = np.fft.ifft(np.fft.fft(u) * kernel_fft)

    return z


def complex_param_perturbated(
    kernel_fft: np.ndarray, phases: np.ndarray, epsh_func: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Compute the perturbed complex order parameter with spatial forcing.

    :param kernel_fft: Precomputed FFT of the coupling kernel
    :param phases: Phase array (length N)
    :param epsh_func: Spatial forcing function epsilon * h(x) (length N)
    :param alpha: Phase lag parameter
    :returns: Perturbed complex order parameter array (length N)
    """
    phase_factors_perturated = np.exp(1j * (phases + alpha + np.pi / 2))
    perturbatred_term = epsh_func * phase_factors_perturated
    z = complex_param(phases, kernel_fft) + perturbatred_term

    return z


def de_phase_solution(theta: np.ndarray) -> np.ndarray:
    """
    Remove global phase drift from phase time series for visualization.

    :param theta: Phase array of shape (time_steps, N)
    :returns: Dephased solution array of shape (time_steps, N) in [0, 2pi)
    """
    Z = np.exp(1j * theta).mean(axis=1)
    psi = np.unwrap(np.angle(Z))
    theta_unwrap = np.unwrap(theta, axis=0)
    theta_detr = theta_unwrap - psi[:, None]
    return theta_detr % (2 * np.pi)
