import numpy as np
from scipy.linalg import toeplitz, solve_toeplitz
import scipy.stats

# Covariance matrices
def get_toeplitz(n, h, memory = "AR1"):
    '''
    Return first line for cov matrix of fractional Gaussian noise or AR process
    - n: size
    - h: parameter (in [0,1])
    - short memory: fGn(False) or AR(True)
    '''
    if memory == "AR1":
        if h==0:
            a = np.zeros(n)
            a[0] = 1
            return a
        return h**(np.arange(n))
    a = np.arange(n)
    return 1/2*((a+1)**(2*h) + (abs(a-1)**(2*h) - 2*a**(2*h)))

def get_cov(n, h, memory = "AR1"):
    '''
    Return covariance matrix of fractional Gaussian noise or AR process
    - n: size
    - h: parameter (in [0,1])
    - short memory: fGn(False) or AR(True)
    '''
    return toeplitz(get_toeplitz(n, h, memory = memory))

def log_det_cov_AR1(n, h):
    return (n-1)*np.log(1-h**2)
    

# Metropolis-Hastings algorithm for H and K

def log_p(params, data, constants, key, x, memory = "AR1"):
    '''
    Compute log-likelihood for H and K knowing all other parameters
    '''
    past, present, future = constants['past'](), constants['present'](), constants['future']()
    if key=='H':
        T = np.concatenate((params['T13']()[:past], data['T2']()))
        T = np.array([np.ones((present)), T])
        M = data['RP']() - np.dot(params['alpha'](), T)
        cov = get_toeplitz(present, x, memory=memory) 
        if memory == "AR1":
            slogdet = log_det_cov_AR1(present, x)
        else:
            slogdet = np.linalg.slogdet(toeplitz(cov))[1]   # log-determinant   
        v = solve_toeplitz(cov, M)
        return -1/2*slogdet-1/(2*params['sigma_p']()**2)*np.inner(M, v)

    if key=='K':
        T = np.concatenate((params['T13']()[:past], data['T2'](), params['T13']()[past:]))
        F = np.array([np.ones((future)), data['S'](), data['V'](), data['C']()]).T
        cov = get_toeplitz(future, x, memory=memory)
        if memory == "AR1":
            slogdet = log_det_cov_AR1(future, x)
        else:
            slogdet = np.linalg.slogdet(toeplitz(cov))[1]   # log-determinant
        M = T - np.dot(params['beta'](), F.T)
        v = solve_toeplitz(cov, M)
        return -1/2*slogdet-1/(2*params['sigma_T']()**2)*np.inner(M, v)
