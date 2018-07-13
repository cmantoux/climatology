import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import AR1
import validation

import numpy as np
from IPython.display import display
from bqplot import (
    OrdinalScale, LinearScale, Lines, Axis, Figure
)
from dashboard import Dashboard
from jinja2 import Template
import xarray as xa

class Gibbs:

    def __init__(self, model, noise_H = AR1(), noise_K=AR1(), dashboard = False):
        '''
        Gibbs sampler class (object):
        - params : dic of parameters with initial values
        - data : data used in simulation
        - laws : dic (same keys as params) -> functions for simulating param knowing the others
            ex: laws[key] = f where:
                
                def f(params, data):
                    ....
                    return np.random.?(???)
        To run Gibbs sampler, use method: Gibbs_object.run()
        To get history (simulated parameters), use Gibbs_object.get_history(key)
        '''
        self.model = model
        self.history = {}  # keep record of simulated parameters
        self.T_check = []

        self.x_ord = OrdinalScale()
        self.y_sc = LinearScale()
        self.x_data = np.array([])
        self.noise_H = noise_H
        self.noise_K = noise_K
        for key in model.params.keys():
            self.history[key] = []
        if dashboard:
            self.dashboard = Dashboard(self.model)
        else:
            self.dashboard = None


    def run(self, n = 100, verbose = False):
        '''
        Run Gibbs sampler:
        - n: number of iterations (default: 100)
        '''
        for k in tqdm(range(n)):
            self.x_data = np.arange(len(self.x_data)+1)
            if verbose:
                print(self.model.params)
            if k % 10 == 0 and self.dashboard is not None:
                self.iteration(update_plot=True)
            else:
                self.iteration()
    
    def iteration(self, update_plot = False):
        self.model.n_iteration += 1
        for key in self.model.params.keys():
            self.simulate(key)
            if update_plot and key not in ['S', 'V', 'C']:
                self.update_plot(key)
    
    def update_plot(self, key):
        template = Template(open('template.html').read())
        fig = self.dashboard.dic_figures[key]
        label = self.dashboard.labels[key]
        hist = self.get_history(key)
        if key == 'T13':
            past, present, future = self.model.constants['past'](), self.model.constants['present'](), self.model.constants['future']()
            T2 = self.model.data['T2']()
            fig.marks[0].x = np.arange(1000, 1000 + future)
            fig.marks[0].y = np.concatenate([hist[-1,:past], T2, hist[-1,past:]])
            return
        for k in range(fig.n):
            fig.marks[k].x = self.x_data
            if fig.n>1:
                fig.marks[k].y = hist[:,k]
            else:
                fig.marks[k].y = hist
        if fig.n <= 1:
            label.value = template.render(inds = range(fig.n), variable = [np.round(np.mean(hist[-10:]),2)], key = key)
        else:
            label.value = template.render(inds = range(fig.n), variable = np.round(np.mean(hist[-10:,:], axis = 0),2), key = key)

    def simulate(self, key):
        var = self.model.params[key]
        if key == 'T13':
            var.value, T = var.law(self.model, noise_H = self.noise_H, noise_K = self.noise_K)
            self.T_check.append(T)
        else:    
            var.value = var.law(self.model, noise_H = self.noise_H, noise_K = self.noise_K)
        self.history[key].append(var.value)

    def result(self, key, last_n = 100):
        hist = self.get_history(key)
        if len(hist.shape) == 1:
            mean = round(np.mean(hist[-last_n:]),3)
            std = round(np.std(hist[-last_n:]),3)
            print('{} = {} +/- {}'.format(key, mean, std))
            return
        for k in range(hist.shape[1]):
            mean = round(np.mean(hist[-last_n:, k]),3)
            std = round(np.std(hist[-last_n:, k]),3)
            print('{}_{} = {} +/- {}'.format(key, k, mean, std))
    
    def get_results(self, keys, last_n = 100):
        for key in keys:
            self.result(key, last_n=last_n)

    def get_history(self, key):
        '''
        Access history for params self.params[key]
        - key : name of the parameter
        '''
        return np.array(self.history[key])
    
    def plot_history(self, key, burnin = 0):
        '''
        Plot history for one parameter:
        - key : name of the parameter
        '''
        h_to_plot = self.history[key]
        try:
            n = h_to_plot.shape[1]
            for k in range(n):
                plt.plot(h_to_plot[burnin:,k], label = '{}_{}'.format(key, k))
        except:
            plt.plot(h_to_plot[burnin:], label = key)
        plt.title(key)
        plt.legend()
        plt.show()
    
    def histogram(self, key, bins = 25, last_n = 100):
        hist = self.get_history(key)[-last_n:]
        print(hist.shape)
        if len(hist.shape) == 1:
            plt.hist(hist, bins = bins, density=True)
            plt.show()
        else:
            n = hist.shape[1]
            k = n//3
            plt.figure(figsize=(9, 3*(k+1)))
            for i in range(n):
                plt.subplot(k+1, 3, i+1)
                plt.hist(hist[:,i], bins = bins, density=True)
            plt.show()
    
    def plot_T_reconstruction(self, last_n = 100, alpha = 95):
        hist_T = self.get_history('T13')[-last_n:,:]
        mean = hist_T.mean(axis=0)
        quantile1 = np.percentile(hist_T, (100-alpha)/2, axis=0)
        quantile2 = np.percentile(hist_T, (100+alpha)/2, axis=0)
        plt.figure(figsize=(15,10))
        t1 = 1000
        past, present, future = self.model.constants['past'](), self.model.constants['present'](), self.model.constants['future']()
        
        plt.plot(np.arange(t1 + past, t1 + present), self.model.data['T2'](), color = 'b', label = 'Known temperatures')
        
        plt.plot(np.arange(t1, t1 + past), mean[:past], color = 'g', label = 'Mean temperatures')
        plt.plot(np.arange(t1 + present ,t1 + future), mean[past:], color = 'g')
        plt.fill_between(np.arange(t1, t1 + past), quantile1[:past], quantile2[:past], color="g", alpha=0.3, label="Q 95%")
        plt.fill_between(np.arange(t1 + present,t1 + future), quantile1[past:], quantile2[past:], color="g", alpha=0.3)


        plt.legend()
        plt.show()

    def save_to_xarray(self, filename = 'dataset.netcdf'):
        RP = xa.DataArray(self.model.data['RP'](), dims=['year'], coords=[np.arange(self.model.t1,self.model.t3)])
        S = xa.DataArray(self.model.data['S'](), dims=['year'], coords=[np.arange(self.model.t1,self.model.t4)])
        V = xa.DataArray(self.model.data['V'](), dims=['year'], coords=[np.arange(self.model.t1,self.model.t4)])
        C = xa.DataArray(self.model.data['C'](), dims=['year'], coords=[np.arange(self.model.t1,self.model.t4)])
        T2 = xa.DataArray(self.model.data['T2'](), dims=['year'], coords=[np.arange(self.model.t2,self.model.t3)])
        
        n = len(self.get_history('sigma_p'))
        alpha = xa.DataArray(self.get_history('alpha'), dims = ['gibbs_it', 'd_alpha'])
        beta = xa.DataArray(self.get_history('beta'), dims = ['gibbs_it', 'd_beta'])
        sigma_p = xa.DataArray(self.get_history('sigma_p'), dims=['gibbs_it'])
        sigma_T = xa.DataArray(self.get_history('sigma_T'), dims=['gibbs_it'])
        if self.noise_H.n_params <= 1:
            H = xa.DataArray(self.get_history('H'), dims = ['gibbs_it'])
        else:
            H = xa.DataArray(self.get_history('H'), dims = ['gibbs_it', 'd_H'])
        if self.noise_K.n_params <= 1:
            K = xa.DataArray(self.get_history('K'), dims = ['gibbs_it'])
        else:
            K = xa.DataArray(self.get_history('K'), dims = ['gibbs_it', 'd_K'])
        T13 = xa.DataArray(self.get_history('T13'), dims = ['gibbs_it', 'year'], coords = [np.arange(n), np.concatenate([np.arange(self.model.t1,self.model.t2), np.arange(self.model.t3,self.model.t4)])])
        T_check = xa.DataArray(np.array(self.T_check), dims = ['gibbs_it', 'year'], coords = [np.arange(n), np.arange(self.model.t2,self.model.t3)])

        Data = xa.Dataset({'RP':RP, 'S':S, 'V':V, 'C':C, 'T2':T2, 'T_check':T_check, 'alpha':alpha, 'beta':beta, 'sigma_p':sigma_p, 'sigma_T':sigma_T, 'H':H, 'K':K, 'T13':T13})
        Data.to_netcdf(filename)
        return Data
    
    def ECP(self, a=0.95, last_n=5000):
        return validation.ECP(self.model.data['T2'](), np.array(self.T_check), a, last_n)
    
    def RMSE(self, last_n=5000):
        return validation.RMSE(self.model.data['T2'](), np.array(self.T_check), last_n)
    
    def CRPS(self, last_n=5000):
        return validation.CRPS(self.model.data['T2'](), np.array(self.T_check), last_n)
    
    def IS(self, a=0.95, last_n=5000):
        return validation.IS(self.model.data['T2'](), np.array(self.T_check), a, last_n)