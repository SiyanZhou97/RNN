import copy
import math
import cupy as cp
import numpy as np


class Rls():
    def __init__(self,M,lr=1):
        """initialize the inv corr matrix P for RLS-based RNN training

        Args:
            M (int): number of source neurons
            N (int, optional): number of target neurons. Needed if reg_ratio is not None
            reg_ratio (scalar,optional):
                apply extra regularization on the [M-N:] columns of the weight, through P0.
                In practice, these are cue & beh feedback weights
            lr (scalar,optional): learning rate
        """
        self.P0=1
        self.M=M
        self.P = self.P0 * cp.eye(self.M, self.M)
        self.lr=lr

    def update(self, weight, input, err):
        """

        Args:
            weight (ndarray of shape (N,M)): weight matrix being optimized
            input (ndarray of shape (M, ))
            err (ndarray of shape (N, ))

        Returns:
            updated weight matrix

        """
        Pu = cp.dot(self.P, input)
        k = Pu / (1 + cp.dot(input, Pu))
        self.P = self.P - cp.outer(k, cp.dot(input, self.P))
        weight = weight - self.lr * cp.outer(err, k)
       
        return weight
