import copy
import math
from tqdm import tqdm_notebook as tqdm
import numpy as np
import cupy as cp

def phi(x, name='tanh', b=None):
    """
    activation function
    Args:
        x (torch tensor): inputs
        b (scaler): offset
    """
    if name == 'tanh':
        return np.tanh(x)
    if name == 'relu':
        return x * (x > 0)
    if name == 'tanh_norm':
        return (1 + np.tanh(x)) / 2
    if name == 'logistic':
        return 1 / (1 + np.exp(-x))
    if name == 'logistic_b':
        return 1 / (1 + np.exp(-x + b))
    
def pVar(data,pred):
    """
    fraction variance explained 
    
    """
    N=data.shape[0]
    T=data.shape[1]
    stdData = cp.std(data.reshape(N*T))
    pVarN = 1 - (cp.linalg.norm(data - pred, 'fro')/(math.sqrt(N*T)*stdData))**2
    return pVarN


def generate_inputWN(n_step,dim_rec,tauWN,ampWN,dt):
    # Additional input
    eta = ampWN * np.random.randn(dim_rec,n_step)
    inputWN = np.zeros((dim_rec,n_step))
    for t in range(1, n_step):
        inputWN[:,t] = inputWN[:,t - 1] + (dt / tauWN) * (-inputWN[:,t - 1] + eta[:,t - 1])
    inputWN = inputWN
    return inputWN

def moving_bump(target_t,dim_rec,t_max_target):
    # Width of bump
    bump_width = 0.035

    # Generate moving bump
    n_t_target = len(target_t)
    rel_idx = np.arange(dim_rec) / dim_rec
    rel_time = target_t / t_max_target
    target_R = np.exp(-(rel_idx[:, None] - rel_time[None, :]) ** 2 /
                        (2 * bump_width ** 2))
    # The target is rates, which have range (0, 1).
    # Shift min and max so that there's no saturation.
    min_r = 0.0
    max_r = 1
    target_R = (max_r - min_r) * target_R + min_r
    return target_R
