import math
from collections.abc import Callable

import numpy as np
from numba import jit


@jit(nopython=True)
def rk_step(
    state: np.ndarray,
    system: Callable[[np.ndarray], np.ndarray],
    dt: float,
) -> np.ndarray:
    """
    Executes a single step of the fourth-order Runge-Kutta (RK4) numerical method.

    This function advances the state of a given system by one time step using the RK4 method.
    The RK4 method involves four intermediate calculations (k1, k2, k3, k4) to estimate the next state of the system.

    
    :param state: The current state vector of the system at time t, denoted as x(t).
                            It should have the shape [dimensionality], where each element
                            represents a state variable (e.g., position, velocity).
    :param system: A function representing the vector field
                                                     of the system, f(x), used to simulate the oscillator's dynamics.
    :param dt: The simulation time step for the RK4 method, representing the increment in time.

    :returns: The estimated next state of the system, x(t + dt), as a numpy array.
    """

    k1 = dt * system(state)
    k2 = dt * system(state + 0.5 * k1)
    k3 = dt * system(state + 0.5 * k2)
    k4 = dt * system(state + k3)
    state_step: np.ndarray = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    next_state: np.ndarray = state + state_step
    return next_state


@jit(nopython=True)
def rk_interval(
    initial_conditions: np.ndarray,
    system: Callable[[np.ndarray], np.ndarray],
    t_final: float,
    dt: float,
) -> np.ndarray:
    """
    Simulates a dynamical system over a specified time interval using the Runge-Kutta method.

    This function uses the fourth-order Runge-Kutta (RK4) method to numerically simulate the evolution of a given system from its initial state to a specified final time.
    It calculates the system's state at discrete time steps, producing a trajectory of states over the time interval.

    :param initial_conditions: The initial state of the system, represented as
                                         a vector of shape [dimensionality]. Each element
                                         corresponds to a state variable.
    :param system: A function representing the system's
                                                     vector field, $f(x)$, defining the dynamics of the system.
    :param t_final: The end time of the simulation period (in seconds).
    :param dt: The time step used for numerical integration, determining the resolution
                    of the simulation.

    :returns: A numpy array of shape [N, dimensionality], where N is the number of simulation steps. Each row represents the system's state at a specific time step.
    """
    num_points: int = math.floor(t_final / dt)
    dimensionality = initial_conditions.shape[0]
    state_vector = np.zeros(shape=(num_points, dimensionality))
    state_vector[0] = initial_conditions
    next_state: np.ndarray = initial_conditions.copy()

    for i in range(num_points - 1):
        next_state = rk_step(next_state, system, dt)
        state_vector[i + 1] = next_state

    return state_vector
