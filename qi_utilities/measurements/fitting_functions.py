import numpy as np

def exp_decay_func(t, tau, amplitude):
    return amplitude * np.exp(-(t / tau))

def cos_func(x, a, b, c, d):
    return a * np.cos(b*x + c) + d

def damped_osc_func(t, tau, frequency, phase, amplitude,
                    oscillation_offset, exponential_offset):
    return amplitude * np.exp(-(t / tau)) * (np.cos(
        2 * np.pi * frequency * t + phase) + oscillation_offset) + exponential_offset