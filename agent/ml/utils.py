# coding: utf-8

def softmax(x):
    max_value = max(x)
    values = np.exp(x - max_value)
    return values / np.sum(values)
