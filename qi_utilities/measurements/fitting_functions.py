import numpy as np

def linear_func(x, a, b):
    return np.polyval([a, b], x) # for f(x) = a*x + b

def exp_decay_func(t, tau, amplitude, offset):
    return amplitude * np.exp(-(t / tau)) + offset

def cos_func(x, a, b, c, d):
    return a * np.cos(b*x + c) + d

def damped_osc_func(t, tau, frequency, phase, amplitude, offset):
    return amplitude * np.exp(-(t / tau)) * np.cos(2 * np.pi * frequency * t + phase) + offset

def double_damped_osc_func(t,
                           tau_1, frequency_1, phase_1, amplitude_1,
                           tau_2, frequency_2, phase_2, amplitude_2, offset):
    if tau_1 > 0:
        cos_1 = amplitude_1 * np.exp(-(t / tau_1)) * np.cos(2 * np.pi * frequency_1 * t + phase_1)
    else:
        cos_1 = np.zeros_like(t)
    if tau_2 > 0:
        cos_2 = amplitude_2 * np.exp(-(t / tau_2)) * np.cos(2 * np.pi * frequency_2 * t + phase_2)
    else:
        cos_2 = np.zeros_like(t)
    return cos_1 + cos_2 + offset