"""
Program: gbm.py
Author: Mike Liang
Date Created: 2024-07-27

Description:

"""

import numpy as np


class GeometricBrownianMotion:
    def __init__(self, S0: float, mu: float, sigma: float, T: int, M: int, N: int):
        """
        Initialize the parameters
        S0: Initial stock price
        mu: Drift coefficient
        sigma: Volatility coefficient
        T: Time to maturity
        M: Number of time steps
        N: Number of simulation paths
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.M = M
        self.N = N

    def generate_paths(self):
        """
        Generate price paths using the GBM model.
        Returns: Array of simulated paths
        """
        dt = self.T / self.M
        paths = np.zeros((self.M + 1, self.N))
        paths[0] = self.S0

        for t in range(1, self.M + 1):
            z = np.random.standard_normal(self.N)
            paths[t] = paths[t - 1] * np.exp(
                (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
            )

        return paths

    def monte_carlo_option_price(self, option_type, K, r):
        """
        Calculates the option price
        option_type: 'call' or 'put'
        K: Strike price
        r: Risk-free rate
        Returns: Estimated option price
        """
        paths = self.generate_paths()
        if option_type.lower() == "call":
            payoffs = np.maximum(paths[-1] - K, 0)  # Terminal St price - K
        elif option_type.lower() == "put":
            payoffs = np.maximum(K - paths[-1], 0)  # K - Terminal St price
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        return np.exp(-r * self.T) * np.mean(payoffs)  # discount back to present value
