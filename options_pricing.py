"""
Program: options_pricing.py
Author: Mike Liang
Date Created: 2024-07-27

Description:

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from gbm import GeometricBrownianMotion

class OptionPricing:
    def __init__(self, S0, K, T, r, sigma, mu, M, N):
        self.S0 = S0    # Initial stock price
        self.K = K      # Strike price
        self.T = T      # Time to maturity
        self.r = r      # Risk-free rate
        self.sigma = sigma  # Volatility
        self.mu = mu    # Drift coefficient
        self.M = M      # Number of time steps
        self.N = N      # Number of simulation paths

    def monte_carlo_simulation(self, option_type):
        gbm = GeometricBrownianMotion(self.S0, self.mu, self.sigma, self.T, self.M, self.N)
        return gbm.monte_carlo_option_price(option_type, self.K, self.r)

    def black_scholes_price(self, option_type):
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if option_type == 'call':
            price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == 'put':
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        return price

    def plot_convergence(self, num_trials):
        call_mc_prices = []
        put_mc_prices = []
        call_bs_price = self.black_scholes_price('call')
        put_bs_price = self.black_scholes_price('put')

        for trials in range(1, num_trials + 1):
            self.N = trials
            call_mc_prices.append(self.monte_carlo_simulation('call'))
            put_mc_prices.append(self.monte_carlo_simulation('put'))

        plt.figure(figsize=(10, 6))
        plt.plot(call_mc_prices, label='call_MC', color='blue')
        plt.plot(put_mc_prices, label='put_MC', color='green')
        plt.axhline(y=call_bs_price, color='red', linestyle='-', label='call_BS')
        plt.axhline(y=put_bs_price, color='magenta', linestyle='-', label='put_BS')
        plt.xlabel('Trials')
        plt.ylabel('Options Price')
        plt.legend()
        plt.title('Price versus number of trials')
        plt.show()

# Example usage
if __name__ == "__main__":
    S0 = 100      # Initial stock price
    K = 100       # Strike price
    T = 1         # Time to maturity
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility
    mu = 0.05     # Drift
    M = 1000      # Time step
    N = 1000      # Underlying simulations

    option_pricing = OptionPricing(S0, K, T, r, sigma, mu, M, N)
    num_trials = 5000
    option_pricing.plot_convergence(num_trials)
