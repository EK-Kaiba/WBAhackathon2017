# coding: utf-8

import numpy as np

def softmax(x):
    max_value = max(x)
    values = np.exp(x - max_value)
    return values / np.sum(values)
