"""
Program: gbm.py
Author: Mike Liang
Date Created: 2024-07-27

Description:

A geometric brownian motion simulation which returns log normal values

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class GBM:
    def __init__(self):
        self.N = int(input("Simulations: "))
        self.T = int(input("Time to maturity (in years): "))
        self.mu = float(input("Drift: "))
        self.sigma = float(input("Volatility: "))
        self.M = input("Time steps: ")
        if not self.M:
            self.M = int(252 * self.T)
        else:
            self.M = int(self.M)
        self.dt = self.T / self.M  # Corrected time step

    def __repr__(self):
        return (
            f"GBM(N={self.N}, T={self.T}, mu={self.mu}, sigma={self.sigma}, "
            f"M={self.M}, dt={self.dt})"
        )

    def log_return(self):
        """
        Simulates log returns for the GBM model.

        :return: An array of shape (N, M) of normally distributed values
        """
        deterministic = self.mu - 0.5 * self.sigma**2
        diffusion = self.sigma

        log_returns = np.random.normal(
            deterministic * self.dt, diffusion * np.sqrt(self.dt), size=(self.N, self.M)
        )

        return log_returns

    def simulate_paths(self):
        """
        Simulates the log return paths for the asset.

        :return: An array of shape (N, M+1) of simulated log return paths
        """
        log_returns = self.log_return()
        returns_vector = np.zeros((self.N, self.M + 1))

        for i in range(1, self.M + 1):
            returns_vector[:, i] = (
                returns_vector[:, i - 1] + log_returns[:, i - 1]
            )  # Sum of log returns

        return returns_vector

    import matplotlib.gridspec as gridspec

    def plot_graph(self, filename="gbm_simulation.png", dpi=600):
        """
        Plots the simulated log return paths and saves the plot as a high-quality PNG file.

        :param filename: The name of the file to save the plot.
        :param dpi: The resolution of the saved plot in dots per inch.
        """
        paths = self.simulate_paths()
        time = np.linspace(0, self.T, self.M + 1)

        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

        ax1 = fig.add_subplot(gs[0])

        for i in range(self.N):
            ax1.plot(time, paths[i], color="black", alpha=0.1)

        # Plot the mean path
        ax1.plot(time, paths.mean(axis=0), color="cyan", linestyle="--")
        ax1.axhline(y=0, color="red", linestyle="--")

        ax1.set_title("GBM simulation of log returns over time")
        ax1.set_xlabel("Time (years)")
        ax1.set_ylabel("Log return")

        # Add text box with parameters
        params_text = (
            f"N (Simulations): {self.N}\n"
            f"T (Years): {self.T}\n"
            f"mu (Drift): {self.mu}\n"
            f"sigma (Volatility): {self.sigma}\n"
            f"M (Time steps): {self.M}"
        )
        ax1.text(
            0.05,
            0.95,
            params_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Create the secondary axis for the distribution plot
        ax2 = fig.add_subplot(gs[1])
        final_log_returns = paths[:, -1]
        ax2.hist(
            final_log_returns,
            bins=30,
            orientation="horizontal",
            color="blue",
            alpha=0.7,
        )
        ax2.set_xlabel("Frequency")
        ax2.set_title("Distribution")

        # Save the plot with a higher dpi setting
        plt.savefig(filename, dpi=dpi)
        plt.show()


gbm = GBM()
print(gbm)
gbm.plot_graph("gbm_simulation.png", 600)
