import numpy as np
from scipy.linalg import solve_toeplitz, toeplitz

def simul_alpha(model, noise):
    '''
    Simulate alpha in Gibbs sampler knowing all other parameters
    '''
    params, data, constants = model.params, model.data, model.constants
    past, present = constants['past'](), constants['present']()

    T = np.concatenate((params['T13']()[:past], data['T2']()))
    T = np.array([np.ones((present)), T])
    cov_top = noise.get_toeplitz(present, params['HK']()[0])
    b = solve_toeplitz(cov_top, T.T)
    P1 = np.dot(T, b)
    P2 = np.dot(data['RP'](), b)
    
    omega = np.linalg.inv(1/params['sigma_p']()**2 * P1 + np.identity(2))
    delta = 1/params['sigma_p']()**2 * P2 + np.array([0,1])
    return np.random.multivariate_normal(mean = np.dot(delta, omega), cov = omega)

def simul_beta(model, noise):
    '''
    Simulate beta in Gibbs sampler knowing all other parameters
    '''
    params, data, constants = model.params, model.data, model.constants
    past, future = constants['past'](), constants['future']()
    T = np.concatenate((params['T13']()[:past], data['T2'](), params['T13']()[past:]))
    F = np.array([np.ones((future)), data['S'](), data['V'](), data['C']()])
    cov_top = noise.get_toeplitz(future, params['HK']()[1])
    b = solve_toeplitz(cov_top, F.T)
    P1 = np.dot(F, b)
    P2 = np.dot(T, b)
    omega = np.linalg.inv(1/params['sigma_T']()**2 * P1 + np.identity(4))
    delta = 1/params['sigma_T']()**2 * P2 + np.array([0,1,1,1])
    return np.random.multivariate_normal(mean = np.dot(delta, omega), cov = omega)

def simul_s_p(model, noise):
    '''
    Simulate sigma_p in Gibbs sampler knowing all other parameters
    '''
    params, data, constants = model.params, model.data, model.constants
    past, present = constants['past'](), constants['present']()
    T = np.concatenate((params['T13']()[:past], data['T2']()))
    T = np.array([np.ones((present)), T])
    cov_top = noise.get_toeplitz(present, params['HK']()[0])
    P1 = np.dot(params['alpha'](), T)
    b = solve_toeplitz(cov_top, data['RP']() - P1)
    P2 = np.dot((data['RP']() - P1).T,b)
    q = 2 + present/2
    r = 0.1 + 1/2 * P2
    return np.sqrt(1/np.random.gamma(shape = q, scale = 1/r))

def simul_s_T(model, noise):
    '''
    Simulate sigma_T in Gibbs sampler knowing all other parameters
    '''
    params, data, constants = model.params, model.data, model.constants
    past, future = constants['past'](), constants['future']()
    T = np.concatenate((params['T13']()[:past], data['T2'](), params['T13']()[past:]))
    F = np.array([np.ones((future)), data['S'](), data['V'](), data['C']()])
    cov_top = noise.get_toeplitz(future, params['HK']()[1])
    P1 = np.dot(params['beta'](), F)
    b = solve_toeplitz(cov_top, T - P1)
    P2 = np.dot((T - P1).T,b)
    q = 2 + future/2
    r = 0.1 + 1/2 * P2
    return np.sqrt(1/np.random.gamma(shape = q, scale = 1/r))

def simul_T(model, noise):
    '''
    Simulate T13 in Gibbs sampler knowing all other parameters
    '''
    params, data, constants = model.params, model.data, model.constants
    past, present, future = constants['past'](), constants['present'](), constants['future']()
    F = np.array([np.ones((future)), data['S'](), data['V'](), data['C']()]).T
    inv_covH = np.pad(np.linalg.inv(noise.get_cov(present, params['HK']()[0])), pad_width=(0,future-present), mode = 'constant')
    inv_covK = np.linalg.inv(noise.get_cov(future, params['HK']()[1]))
    
    P1 = np.dot(inv_covH, np.pad(data['RP'](), pad_width=(0,future-present), mode = 'constant') - params['alpha']()[0])
    P2 = np.dot(inv_covK, np.dot(F, params['beta']()))
    omega = np.linalg.inv((params['alpha']()[1]/params['sigma_p']())**2*inv_covH + 1/params['sigma_T']()**2 * inv_covK)
    delta = params['alpha']()[1]/params['sigma_p']()**2 * P1 + 1/params['sigma_T']()**2 * P2
    mu = np.dot(delta, omega)
    # T = mu + np.dot(np.linalg.cholesky(omega), np.random.randn(len(mu)))
    
    
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

def simul_H_K(model, noise):
    '''
    Simulate H and K in Gibbs sampler knowing all other parameters
    '''
    
    params, data, constants = model.params, model.data, model.constants
    HK = model.params['HK']()
    H, K = HK[0], HK[1]
    
    step_H, step_K = constants['step_H'](), constants['step_K']()
    n_iteration = constants['n_iteration']()

    # Simulation of H
    acc = 0
    for k in range(n_iteration):
        new_H = noise.draw_MH(H, step_H)
        log_p1 = noise.log_p(params, data, constants, 'H', new_H)
        log_p2 = noise.log_p(params, data, constants, 'H', H)
        log_q1 = noise.log_q(new_H, step_H)
        log_q2 = noise.log_q(H, step_H)
        alpha = log_p1 + log_q1 - log_p2 - log_q2
        a = np.random.uniform()
        if np.log(a) <= alpha: # Accept
            H = new_H
            acc += 1

    # Simulation of K
    acc = 0
    for k in range(n_iteration):
        new_K = noise.draw_MH(K, step_K)
        log_p1 = noise.log_p(params, data, constants, 'K', new_K)
        log_p2 = noise.log_p(params, data, constants, 'K', K)
        log_q1 = noise.log_q(new_K, step_K)
        log_q2 = noise.log_q(K, step_K)
        alpha = log_p1 + log_q1 - log_p2 - log_q2
        a = np.random.uniform()
        if np.log(a) <= alpha: # Accept
            K = K = new_K
            acc += 1
    return np.array([H, K])
    

