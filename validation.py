"""Gather the various validation metrics used to assess the quality of a Bayesian model.
    All the metrics take a Gibbs object and return a number"""

import numpy as np

def ECP(real_series, simulated_series, a=0.95, last_n=5000):
    """
    Empirical Coverage Probability for the confidence interval :
    Computes the proportion of real temperatures (ie. in T2) falling in the
    empirical confidence interval with level alpha
    Best value : alpha
    """
    T2 = real_series
    T_check = simulated_series[-last_n:, :]
    q1 = np.percentile(T_check, (1-a)/2*100, axis=0)
    q3 = np.percentile(T_check, (1+a)/2*100, axis=0)
    return np.mean((T2<=q3)*(T2>=q1))

def RMSE(real_series, simulated_series, last_n):
    """
    Root Mean Square Error of the model.
    Best value : 0
    """
    T2 = real_series
    T_check = simulated_series[-last_n:, :]
    E = T2-T_check
    SE = np.power(E, 2)
    MSE = np.mean(SE, axis=0)
    RMSE_res = np.sqrt(MSE)
    return np.mean(RMSE_res)

def CRPS(real_series, simulated_series, last_n):
    """
    Continuous Rank Probability Score, computed using the empirical cumulative distribution function.
    \int_R (F_X(y)-1_{y>x})^2 dy
    See Barboza online supplement A.3 for implementation details. (average the CRPS over time)
    Best value : 0
    """
    last_n -= last_n % 2 # We make last_n even
    T2 = real_series
    T_check = simulated_series[-last_n:, :]
    T_check_first_half = T_check[:last_n//2, :]
    T_check_second_half = T_check[last_n//2:, :]
    CRPS_res = np.mean(np.abs(T_check_second_half-T2)-np.abs(T_check_second_half-T_check_first_half)/2)
    
    res = 0
    for t in range(len(T2)):
        res_tmp = 0
        for i in range(last_n//2):
            res_tmp += abs(T_check[last_n//2+i][t]-T2[t])-0.5*abs(T_check[last_n//2+i][t]-T_check[i][t])
        res_tmp /= last_n//2
        res += res_tmp
    res /= len(T2)
    return CRPS_res, res

def IS(real_series, simulated_series, a=0.95, last_n=5000):
    """
    Interval Score
    Best value : 0
    """
    a_c = 1-a
    T2 = real_series
    T_check = simulated_series[-last_n:, :]
    
    q1 = np.percentile(T_check, (a_c/2)*100, axis=0)
    q3 = np.percentile(T_check, (1-a_c/2)*100, axis=0)
    IS_res = []
    for i in range(len(T2)):
        if T2[i] < q1[i]:
            IS_res.append(2*a_c + 4*(q1[i]-T2[i]))
        elif T2[i] > q3[i]:
            IS_res.append(2*a_c + 4*(T2[i]-q3[i]))
        else:
            IS_res.append(2*a_c)
    return np.mean(IS_res)