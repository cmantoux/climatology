import numpy as np
from scipy.linalg import toeplitz, solve_toeplitz
from scipy.stats import truncnorm

class Noise:
    def __init__(self):
        self.n_params = 0
        self.name = "-"

    def get_toeplitz(self, n, h):
        # The first row of the covariance matrix
        raise NotImplementedError()

    def get_cov(self, n, h):
        # The full covariance matrix
        return toeplitz(self.get_toeplitz(n, h))

    def get_logdet(self, n, h):
        # The logarithm of the determinant of the covariance matrix
        cov = self.get_toeplitz(n, h)
        return np.linalg.slogdet(toeplitz(cov))[1]   # log-determinant

    def log_p(self, params, data, constants, key, x):
        # Computes the log-likelihood for H and K knowing all other parameters
        past, present, future = constants['past'](), constants['present'](), constants['future']()
        if key=='H':
            T = np.concatenate((params['T13']()[:past], data['T2']()))
            T = np.array([np.ones((present)), T])
            M = data['RP']() - np.dot(params['alpha'](), T)
            cov = self.get_toeplitz(present, x)
            slogdet = self.get_logdet(present, x)
            v = solve_toeplitz(cov, M)
            return -1/2*slogdet-1/(2*params['sigma_p']()**2)*np.inner(M, v)

        if key=='K':
            T = np.concatenate((params['T13']()[:past], data['T2'](), params['T13']()[past:]))
            F = np.array([np.ones((future)), data['S'](), data['V'](), data['C']()]).T
            cov = self.get_toeplitz(future, x)
            slogdet = self.get_logdet(future, x)
            M = T - np.dot(params['beta'](), F.T)
            v = solve_toeplitz(cov, M)
            return -1/2*slogdet-1/(2*params['sigma_T']()**2)*np.inner(M, v)

    def draw_MH(self):
        # Draws one sample of the H parameter for the Metropolis-Hastings algorithm
        raise NotImplementedError()

    def log_q(self, h, new_h, step_h):
        # Computes the probability density function for the transition kernel of the Metropolis-Hastings algorithm
        raise NotImplementedError()

class WhiteNoise(Noise):
    def __init__(self):
        self.n_params = 0
        self.name = "WN"

    def get_toeplitz(self, n, h):
        a = np.zeros(n)
        a[0] = 1
        return a

class FractionalGaussianNoise(Noise):
    def __init__(self):
        self.n_params = 1
        self.name = "fGn"
        self.lower = 0
        self.upper = 1

    def get_toeplitz(self, n, h):
        a = np.arange(n)
        return 1/2*((a+1)**(2*h) + (abs(a-1)**(2*h) - 2*a**(2*h)))

    def draw_MH(self, h, step_h):
        return truncnorm.rvs((self.lower - h)/step_h, (self.upper-h)/step_h, loc = h, scale = step_h)

    def log_q(self, h, new_h, step_h):
        return truncnorm.logpdf(new_h, (self.lower - h)/step_h, (self.upper-h)/step_h, loc = h, scale = step_h)


class AR1(Noise):
    # X_{n+1} = H*X_n + e_n
    def __init__(self):
        self.n_params = 1
        self.name = "AR1"
        self.lower = -1
        self.upper = 1

    def get_toeplitz(self, n, h):
        if h==0:
            a = np.zeros(n)
            a[0] = 1
            return a
        return h**(np.arange(n))

    def get_logdet(self, n, h):
        return (n-1)*np.log(1-h**2)

    def draw_MH(self, h, step_h):
        return truncnorm.rvs((self.lower - h)/step_h, (self.upper-h)/step_h, loc = h, scale = step_h)

    def log_q(self, h, new_h, step_h):
        return truncnorm.logpdf(new_h, (self.lower - h)/step_h, (self.upper-h)/step_h, loc = h, scale = step_h)

class AR2(Noise):
    # X_{n+1} = H[0]*X_n + H[1]*X_{n-1} + e_n
    def __init__(self):
        self.n_params = 2
        self.name = "AR2"
        self.lower = -2
        self.upper = 2

    def is_valid(self, h):
        f1, f2 = h[0], h[1]
        return np.max(abs(np.roots([1, -f1, -f2]))) < 1-1e-10

    def get_toeplitz(self, n, h):
        f1, f2 = h[0], h[1]
        if not self.is_valid(h):
            raise AssertionError("The AR noise is not stationary for the value H = [{}, {}].".format(f1, f2))
        r = np.zeros(n)
        r[0] = 1
        r[1] = f1/(1-f2)
        i = 2
        while i < n:
            r[i] = f1*r[i-1] + f2*r[i-2]
            i += 1
        return r

    def draw_MH(self, h, step_h):
        # The two parameters of the AR(2) are assumed to be independent
        f1, f2 = h[0], h[1]
        new_f1 = truncnorm.rvs((self.lower - f1)/step_h, (self.upper-f1)/step_h, loc = f1, scale = step_h)
        new_f2 = truncnorm.rvs((self.lower - f2)/step_h, (self.upper-f2)/step_h, loc = f2, scale = step_h)
        while not self.is_valid([new_f1, new_f2]):
            new_f1 = truncnorm.rvs((self.lower - f1)/step_h, (self.upper-f1)/step_h, loc = f1, scale = step_h)
            new_f2 = truncnorm.rvs((self.lower - f2)/step_h, (self.upper-f2)/step_h, loc = f2, scale = step_h)
        return np.array([new_f1, new_f2])

    def log_q(self, h, new_h, step_h):
        f1, f2 = h[0], h[1]
        new_f1, new_f2 = new_h[0], new_h[1]
        log_q1 = truncnorm.logpdf(new_f1, (self.lower - f1)/step_h, (self.upper-f1)/step_h, loc = f1, scale = step_h)
        log_q2 = truncnorm.logpdf(new_f2, (self.lower - f2)/step_h, (self.upper-f2)/step_h, loc = f2, scale = step_h)
        return log_q1 + log_q2
