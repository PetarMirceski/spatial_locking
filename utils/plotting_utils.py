import math

import numpy as np
from scipy.integrate._ivp.ivp import OdeResult


def generate_yticks_and_labels(
    sol: OdeResult, evolve_time: float, every_n: int = 100
) -> tuple[list[int], list[str]]:
    """
    Generates yticks and corresponding labels based on a solution object's time array.
    The function always rounds up to the neaerest 100.
    Use the function if you want to plot the solution at specific time points of 100s or 1000s.

    :param: sol: A solution object with a `t` attribute (a NumPy array) representing time points.
    :param: evolve_time: Total evolution time for the simulation.

    :returns: A tuple containing:
    - yticks: List of indices corresponding to closest time points.
    - ytick_labels: List of rounded-up time labels.
    """

    def roundup(x: float) -> int:
        """Round up a float to the nearest n-th."""
        return math.ceil(x / 100.0) * 100

    yticks: list[int] = []
    ytick_labels: list[str] = []

    for time in np.arange(0, evolve_time + 1, every_n):
        # Find the index of the closest time point
        idx = int((np.abs(sol.t - time)).argmin())
        yticks.append(idx)
        ytick_labels.append(str(roundup(sol.t[idx])))

    return yticks, ytick_labels
