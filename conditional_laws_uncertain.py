import numpy as np
from scipy.linalg import solve_toeplitz, toeplitz

def simul_beta3(model, noise_H, noise_K):
    '''
    Simulate beta in Gibbs sampler knowing all other parameters
    '''
    params, data, constants = model.params, model.data, model.constants
    # return params['beta']()
    past, future = constants['past'](), constants['future']()
    T = np.concatenate((params['T13']()[:past], data['T2'](), params['T13']()[past:]))
    cov_top = noise_K.get_toeplitz(future, params['K']())
    M = np.tril(toeplitz(cov_top))
    cov_top = cov_top / (1-params['K']()**2)
    S = params['S']()
    V = params['V']()
    C = params['C']()
    v = np.array([np.dot(M, np.ones((future))), np.dot(M,S), np.dot(M,V), np.dot(M,C)])
    b = solve_toeplitz(cov_top, v.T)
    
    P1 = np.dot(v, b)
    
    omega = np.linalg.inv(1/params['sigma_T']()**2 * P1 + np.identity(4))
    mu = (1/params['sigma_T']()**2)*omega.dot(T.dot(b))
    a = np.random.multivariate_normal(mean = mu, cov = omega)
    return a

def simul_s_T3(model, noise_H, noise_K):
    '''
    Simulate sigma_T in Gibbs sampler knowing all other parameters
    '''
    params, data, constants = model.params, model.data, model.constants
    # return params['sigma_T']()
    past, future = constants['past'](), constants['future']()
    T = np.concatenate((params['T13']()[:past], data['T2'](), params['T13']()[past:]))
    cov_top = noise_K.get_toeplitz(future, params['K']())
    M = np.tril(toeplitz(cov_top))
    cov_top = cov_top / (1-params['K']()**2)
    S = params['S']()
    V = params['V']()
    C = params['C']()
    v = np.array([np.dot(M, np.ones((future))), np.dot(M,S), np.dot(M,V), np.dot(M,C)])
    P1 = np.dot(params['beta'](), v)
    b = solve_toeplitz(cov_top, T - P1)
    P2 = np.dot((T - P1).T,b)
    q = 2 + future/2
    r = 0.1 + 1/2 * P2
    return np.sqrt(1/np.random.gamma(shape = q, scale = 1/r))

def simul_T3(model, noise_H, noise_K):
    '''
    Simulate T13 in Gibbs sampler knowing all other parameters
    '''
    params, data, constants = model.params, model.data, model.constants
    past, present, future = constants['past'](), constants['present'](), constants['future']()
    cov_top_H = noise_K.get_toeplitz(present, params['H']())
    cov_top_K = noise_K.get_toeplitz(future, params['K']())
    M_H = np.tril(toeplitz(cov_top_H))
    M_K = np.tril(toeplitz(cov_top_K))
    cov_top_H = cov_top_H / (1-params['H']()**2)
    cov_top_K = cov_top_K / (1-params['K']()**2)
    S = params['S']()
    V = params['V']()
    C = params['C']()
    v = np.array([np.dot(M_K, np.ones((future))), np.dot(M_K,S), np.dot(M_K,V), np.dot(M_K,C)])

    inv_covK = np.linalg.inv(toeplitz(cov_top_K))
    inv_covH = np.linalg.inv(toeplitz(cov_top_H))

    P1 = np.pad(np.dot(M_H.T, np.dot(inv_covH, M_H)), pad_width = (0,future-present), mode='constant')
    omega = np.linalg.inv((params['alpha']()[1]/params['sigma_p']())**2*P1 + 1/params['sigma_T']()**2 * inv_covK)
    a = solve_toeplitz(cov_top_K, np.dot(params['beta'](),v))
    b = np.pad(np.dot(M_H.T, solve_toeplitz(cov_top_H, data['RP']() - params['alpha']()[0]*np.dot(M_H, np.ones(present)))), pad_width=(0,future-present), mode='constant')
    mu = 1/params['sigma_T']()**2*np.dot(omega, a) + params['alpha']()[1]/params['sigma_p']()**2*np.dot(omega, b)
    # T = mu + np.dot(np.linalg.cholesky(omega), np.random.randn(len(mu)))
    # print(params['alpha']()[1]/params['sigma_p']()**2*np.dot(omega, b))

    M1 = omega[past:present, past:present]
    M1_inv = np.linalg.inv(M1)
    M2 = np.concatenate((omega[:past, past:present], omega[present:future, past:present]), axis = 0)
    M3 = np.concatenate((np.concatenate((omega[:past,:past],omega[:past, present:future]), axis = 1), np.concatenate((omega[present:future,:past], omega[present:future,present:future]), axis = 1)), axis = 0)
    M3_inv = np.linalg.inv(M3)

    # Bloc 1-3
    new_mean13 = np.concatenate((mu[:past], mu[present:future])) + np.dot(M2,np.dot(M1_inv, data['T2']() - mu[past:present]))
    new_cov13 = M3 - np.dot(M2, np.dot(M1_inv, M2.T))
    T13 = new_mean13 + np.dot(np.linalg.cholesky(new_cov13), np.random.randn(len(new_mean13)))
    # Bloc 2
    new_mean2 = mu[past:present] + np.dot(M2.T,np.dot(M3_inv, T13 - np.concatenate((mu[:past], mu[present:future]))))
    new_cov2 = M1 - np.dot(M2.T, np.dot(M3_inv, M2))
    T2 = new_mean2 + np.dot(np.linalg.cholesky(new_cov2), np.random.randn(len(new_mean2)))
    return T13, T2

def simul_S3(model, noise_H, noise_K):
    return model.params['S']()

def simul_V3(model, noise_H, noise_K):
    '''
    The volcanism is modeled as a Gamma process
    '''
    
    params, data, constants = model.params, model.data, model.constants
    past, present, future = constants['past'](), constants['present'](), constants['future']()
    
    spike_0 = present-8 # = 992 = Date of the last spike
    # The first spike has to be in the future
    first_spike = spike_0 + int(np.random.gamma(2.16, 1/0.2))
    while first_spike < present:
        first_spike = spike_0 + int(np.random.gamma(2.16, 1/0.2))
    first_amplitude = np.random.gamma(1.08, 1/19.5)
    spikes = [first_spike]
    amplitudes = [first_amplitude]
    while True:
        spike = spikes[-1] + int(np.random.gamma(2.16, 1/0.2))
        if spike < future:
            spikes.append(spike)
            amplitudes.append(np.random.gamma(1.08, 1/19.5))
        else:
            break
    
    V_spike = data['V_spike']()
    V_cst = data['V_cst']()
    
    V = np.zeros_like(V_cst)+np.min(V_cst)
    V[:present] = V_cst[:present]

    for i in range(len(spikes)):
        t_min = spikes[i]-1
        t_max = min(spikes[i]+8, future)
        V[t_min:t_max] = V[t_min:t_max] + amplitudes[i] * V_spike[:(t_max-t_min)]
    V = np.log(1-V)
    return (V - V_cst[:present].mean())/V_cst[:present].std()

def simul_C3(model, noise_H, noise_K):
    '''
    The CO2 is modeled as a multiplicative random variation of the standard RCPs
    C(1+lambda_t*epsilon) with lambda_t increasing and epsilon Gaussian
    '''
    params, data, constants = model.params, model.data, model.constants
    past, present, future = constants['past'](), constants['present'](), constants['future']()
    
    eps = np.random.randn()
    C_var = data['C_var']()
    C_cst = data['C_cst']()
    variation = np.zeros(future)
    variation[present:] = eps*C_var*np.linspace(0,1,future-present)
    
    return (C_cst + variation - C_cst[:present].mean())/C_cst[:present].std()