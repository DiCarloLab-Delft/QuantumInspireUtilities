import numpy as np

def exp_decay_func(t, tau, amplitude):
    return amplitude * np.exp(-(t / tau))

def cos_func(x, a, b, c, d):
    return a * np.cos(b*x + c) + d