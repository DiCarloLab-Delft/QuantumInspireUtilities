import numpy as np

def exp_decay_func(t, tau, amplitude):
    return amplitude * np.exp(-(t / tau))