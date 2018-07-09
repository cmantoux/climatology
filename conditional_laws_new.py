import numpy as np
from scipy.linalg import solve_toeplitz, toeplitz

def simul_alpha2(model, noise_H, noise_K):
    '''
    Simulate alpha in Gibbs sampler knowing all other parameters
    '''
    params, data, constants = model.params, model.data, model.constants
    # return params['alpha']()
    past, present = constants['past'](), constants['present']()
    # print(params['T13']())
    T = np.concatenate((params['T13']()[:past], data['T2']()))
    RP = data['RP']()
    cov_top = noise_H.get_toeplitz(present, params['H']())
    M = np.tril(toeplitz(cov_top))
    cov_top = cov_top / (1-params['H']()**2)
    u = np.array([np.dot(M,np.ones(len(T))), np.dot(M,T)])
    b = solve_toeplitz(cov_top, u.T)

    P1 = np.dot(u,b)

    omega = np.linalg.inv(1/params['sigma_p']()**2 *P1 + np.identity(2))
    c = solve_toeplitz(cov_top, RP)
    mu = 1/params['sigma_p']()**2 *np.dot(np.dot(omega, u), c)
    a = np.random.multivariate_normal(mean = mu, cov = omega)
    return a

def simul_beta2(model, noise_H, noise_K):
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
    S = data['S']()
    V = data['V']()
    C = data['C']()
    v = np.array([np.dot(M, np.ones((future))), np.dot(M,S), np.dot(M,V), np.dot(M,C)])
    b = solve_toeplitz(cov_top, v.T)
    
    P1 = np.dot(v, b)
    
    omega = np.linalg.inv(1/params['sigma_T']()**2 * P1 + np.identity(4))
    mu = (1/params['sigma_T']()**2)*omega.dot(T.dot(b))
    a = np.random.multivariate_normal(mean = mu, cov = omega)
    return a

def simul_s_p2(model, noise_H, noise_K):
    '''
    Simulate sigma_p in Gibbs sampler knowing all other parameters
    '''
    params, data, constants = model.params, model.data, model.constants
    # return params['sigma_p']()
    past, present = constants['past'](), constants['present']()
    T = np.concatenate((params['T13']()[:past], data['T2']()))

    cov_top = noise_H.get_toeplitz(present, params['H']())
    M = np.tril(toeplitz(cov_top))
    cov_top = cov_top / (1-params['H']()**2)
    u = np.array([np.dot(M,np.ones(len(T))), np.dot(M,T)])

    P1 = np.dot(params['alpha'](), u)
    b = solve_toeplitz(cov_top, data['RP']() - P1)
    P2 = np.dot((data['RP']() - P1).T,b)
    q = 2 + present/2
    r = 0.1 + 1/2 * P2
    return np.sqrt(1/np.random.gamma(shape = q, scale = 1/r))

def simul_s_T2(model, noise_H, noise_K):
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
    S = data['S']()
    V = data['V']()
    C = data['C']()
    v = np.array([np.dot(M, np.ones((future))), np.dot(M,S), np.dot(M,V), np.dot(M,C)])
    P1 = np.dot(params['beta'](), v)
    b = solve_toeplitz(cov_top, T - P1)
    P2 = np.dot((T - P1).T,b)
    q = 2 + future/2
    r = 0.1 + 1/2 * P2
    return np.sqrt(1/np.random.gamma(shape = q, scale = 1/r))

def simul_T2(model, noise_H, noise_K):
    '''
    Simulate T13 in Gibbs sampler knowing all other parameters
    '''
    params, data, constants = model.params, model.data, model.constants
    # return params['T13'](), data['T2']()
    past, present, future = constants['past'](), constants['present'](), constants['future']()
    cov_top_H = noise_K.get_toeplitz(present, params['H']())
    cov_top_K = noise_K.get_toeplitz(future, params['K']())
    M_H = np.tril(toeplitz(cov_top_H))
    M_K = np.tril(toeplitz(cov_top_K))
    cov_top_H = cov_top_H / (1-params['H']()**2)
    cov_top_K = cov_top_K / (1-params['K']()**2)
    S = data['S']()
    V = data['V']()
    C = data['C']()
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

def simul_H2(model, noise_H, noise_K):
    '''
    Simulate H in Gibbs sampler knowing all other parameters
    '''

    params, data, constants = model.params, model.data, model.constants
    H = model.params['H']()
    return H

    if noise_H.n_params == 0:
        return H

    step_H = constants['step_H']()
    n_iteration = constants['n_iteration']()

    # Simulation of H
    acc = 0
    for k in range(n_iteration):
        new_H = noise_H.draw_MH(H, step_H)
        log_p1 = noise_H.log_p2(params, data, constants, 'H', new_H)
        log_p2 = noise_H.log_p2(params, data, constants, 'H', H)
        log_q1 = noise_H.log_q(H, new_H, step_H)
        log_q2 = noise_H.log_q(new_H, H, step_H)
        alpha = log_p1 + log_q1 - log_p2 - log_q2
        a = np.random.uniform()
        if np.log(a) <= alpha: # Accept
            H = new_H
            acc += 1


    return H

def simul_K2(model, noise_H, noise_K):
    '''
    Simulate K in Gibbs sampler knowing all other parameters
    '''

    params, data, constants = model.params, model.data, model.constants
    K = model.params['K']()
    return K
    
    if noise_K.n_params == 0:
        return K

    step_K = constants['step_K']()
    n_iteration = constants['n_iteration']()

    # Simulation of K
    acc = 0
    for k in range(n_iteration):
        new_K = noise_K.draw_MH(K, step_K)
        log_p1 = noise_K.log_p2(params, data, constants, 'K', new_K)
        log_p2 = noise_K.log_p2(params, data, constants, 'K', K)
        log_q1 = noise_K.log_q(K, new_K, step_K)
        log_q2 = noise_K.log_q(new_K, K, step_K)
        alpha = log_p1 + log_q1 - log_p2 - log_q2
        a = np.random.uniform()
        if np.log(a) <= alpha: # Accept
            K = K = new_K
            acc += 1
    return K
