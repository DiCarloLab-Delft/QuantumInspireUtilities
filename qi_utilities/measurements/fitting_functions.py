"""
Model functions used for fitting the measurement data of the
characterization experiments implemented in measurement_classes.py.

Authors: Marios Samiotis
"""

import numpy as np

def linear_func(x, a, b):
    """
    This function returns the line f(x) = a*x + b.

    Args:
        x (float | np.ndarray):
            The independent variable.

        a (float):
            The slope of the line.

        b (float):
            The offset of the line.
    """
    return np.polyval([a, b], x) # for f(x) = a*x + b

def exp_decay_func(t, tau, amplitude, offset):
    """
    This function returns the exponential decay
    f(t) = amplitude * exp(-t/tau) + offset.

    Args:
        t (float | np.ndarray):
            The time expressed in units of seconds [s].

        tau (float):
            The decay time constant expressed in units of seconds [s],
            e.g. the qubit relaxation time T1.

        amplitude (float):
            The amplitude of the exponential decay.

        offset (float):
            The asymptotic value of the exponential decay for t >> tau.
    """
    return amplitude * np.exp(-(t / tau)) + offset

def cos_func(x, amplitude, frequency, phase, offset):
    """
    This function returns the cosine
    f(x) = amplitude * cos(frequency*x + phase) + offset.

    Note: here 'frequency' is an angular frequency expressed in units
    of [rad] per unit of x, meaning there is no factor of 2*pi.
    e.g. for a Rabi oscillation with x the rotation angle in [rad],
    frequency = 1 for a correctly calibrated pi pulse.

    Args:
        x (float | np.ndarray):
            The independent variable, e.g. the rotation angle in [rad].

        amplitude (float):
            The amplitude of the oscillation.

        frequency (float):
            The angular frequency of the oscillation.

        phase (float):
            The phase of the oscillation expressed in [rad].

        offset (float):
            The offset of the oscillation.
    """
    return amplitude * np.cos(frequency*x + phase) + offset

def damped_osc_func(t, tau, frequency, phase, amplitude, offset):
    """
    This function returns the exponentially damped oscillation
    f(t) = amplitude * exp(-t/tau) * cos(2*pi*frequency*t + phase) + offset.

    Args:
        t (float | np.ndarray):
            The time expressed in units of seconds [s].

        tau (float):
            The decay time constant expressed in units of seconds [s],
            e.g. the qubit dephasing time T2*.

        frequency (float):
            The oscillation frequency expressed in units of [Hz].

        phase (float):
            The phase of the oscillation expressed in [rad].

        amplitude (float):
            The amplitude of the oscillation at t = 0.

        offset (float):
            The offset of the oscillation.
    """
    return amplitude * np.exp(-(t / tau)) * np.cos(2 * np.pi * frequency * t + phase) + offset

def double_damped_osc_func(t,
                           tau_1, frequency_1, phase_1, amplitude_1,
                           tau_2, frequency_2, phase_2, amplitude_2, offset):
    """
    This function returns the sum of two exponentially damped oscillations
    (see damped_osc_func) sharing a common offset. It is used for detecting
    a double beating in Ramsey-type experiments, e.g. due to a qubit frequency
    switching between two values during the measurement.

    A term with a non-positive decay time constant (tau_i <= 0) is set to zero.

    Args:
        t (float | np.ndarray):
            The time expressed in units of seconds [s].

        tau_1, tau_2 (float):
            The decay time constants of each oscillation expressed in units
            of seconds [s].

        frequency_1, frequency_2 (float):
            The frequencies of each oscillation expressed in units of [Hz].

        phase_1, phase_2 (float):
            The phases of each oscillation expressed in [rad].

        amplitude_1, amplitude_2 (float):
            The amplitudes of each oscillation at t = 0.

        offset (float):
            The common offset of the two oscillations.
    """
    if tau_1 > 0:
        cos_1 = amplitude_1 * np.exp(-(t / tau_1)) * np.cos(2 * np.pi * frequency_1 * t + phase_1)
    else:
        cos_1 = np.zeros_like(t)
    if tau_2 > 0:
        cos_2 = amplitude_2 * np.exp(-(t / tau_2)) * np.cos(2 * np.pi * frequency_2 * t + phase_2)
    else:
        cos_2 = np.zeros_like(t)
    return cos_1 + cos_2 + offset