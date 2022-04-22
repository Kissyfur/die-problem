import numpy as np
import math
import algorithms as a
from scipy.special import logsumexp
# from scipy.misc import logsumexp


def entropy(beta):
    p = a.to_p(beta)
    nz = np.nonzero(p)
    return -np.sum(p[nz] * np.log(p[nz]))


def max_entropy(die_dim):
    return np.log(die_dim)


def measure(beta):
    max_beta = np.max(beta)
    p_hat = np.exp(beta - max_beta)
    return np.sum(p_hat)*math.exp(max_beta)


class Measure:
    def __init__(self, name, function):
        self.name = name
        self.compute = function


def kl_divergence(beta_real, betas):
    p_real = a.to_p(beta_real)
    qs = a.to_ps(betas)

    non_zeros = np.nonzero(p_real)
    kl = -p_real[non_zeros] * np.log(qs.T[non_zeros]).T + p_real[non_zeros] * np.log(p_real)[non_zeros]
    kl = np.sum(kl, axis=1)
    return kl

def klpq(real_beta, beta, y):
    p_real = a.to_p(real_beta)
    p = a.to_p(beta)
    non_zeros = (p_real!=0)
    return np.sum(p_real[non_zeros]*(np.log(p_real)[non_zeros] - np.log(p)[non_zeros]))


KLpq = Measure("KLpq", klpq)

def klqp(real_beta, beta, y):
    return klpq(beta, real_beta, y)
#    p_real = to_p(real_beta)
#    p = to_p(beta)
#    non_zeros = (p!=0)
#    return np.sum(p[non_zeros]*(np.log(p)[non_zeros] - np.log(p_real)[non_zeros]))


KLqp = Measure("KLqp", klqp)


def logs_p(real_beta, betas, y):
    lse = logsumexp(betas, axis=1)
    return np.sum(betas.T[y].T - lse, axis=1)/len(y)


def log_p(real_beta, beta, y):
    lse = logsumexp(beta)
    return np.sum(beta[y] - lse)/len(y)


LogLikelihood = Measure("LL", log_p)
