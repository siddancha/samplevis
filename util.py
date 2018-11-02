import numpy as np
from easydict import EasyDict as edict

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def inverse_sigmoid(x):
    return np.log(x) - np.log(1. - x)

def merge_dicts(dict_user, dict_default):
    """
    Merges dict_user with dict_default. If a key in dict_default does not exist in
    dict_user, adds the corresponding (key, value) pair to dict_user.
    """
    # Check if each key in dictionary specified by user is valid.
    assert (set(dict_user) <= set(dict_default))

    for key, value in dict_default.items():
        if key not in dict_user:
            dict_user[key] = value

    return edict(dict_user)
