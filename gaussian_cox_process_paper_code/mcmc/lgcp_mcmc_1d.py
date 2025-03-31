import warnings
from itertools import product

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

from matplotlib import MatplotlibDeprecationWarning


warnings.filterwarnings(action="ignore", category=MatplotlibDeprecationWarning)


if __name__ == "__main__":
    run_new_samples = True

    # redwood
    # datasets_2d = ["redwood_full", "white_oak"]
    prior_uppers = [20, 1, 40]
    prior_lowers = [0, 0, 0]
    boundary_uppers = [50, 5, 100]
    boundary_lowers = [0, 0, 0]

    for i in range(1, 4):
        data = pd.read_csv("../datasets/synth{}/observation0.csv".format(i))
        # n = data.shape[0]
        prior_upper = prior_uppers[i - 1]
        prior_lower = prior_lowers[i - 1]
        boundary_upper = boundary_uppers[i - 1]
        boundary_lower = boundary_lowers[i - 1]

        x = data.values

        resolution = 0.9
        area_per_cell = resolution
        grid_size = boundary_upper - boundary_lower
        cells = int(grid_size / resolution)
        quadrat = np.linspace(0, grid_size, cells + 1)
        centroids = np.array(
            quadrat[:-1] + resolution / 2,
        )
        centroids = np.expand_dims(centroids, 1)

        cell_counts, _ = np.histogram(x, bins=quadrat)
        cell_counts = cell_counts.ravel().astype(int)

        line_kwargs = {"color": "k", "linewidth": 1, "alpha": 0.5}

        # "Inference"
        with pm.Model() as lgcp_model:
            mu = pm.Normal("mu", sigma=3)
            rho = pm.Uniform("rho", lower=prior_lower, upper=prior_upper)
            variance = pm.InverseGamma("variance", alpha=1, beta=1)

            # second layer of the hierarchical
            cov_func = variance * pm.gp.cov.Matern52(
                1, ls=rho
            )  # 1 for 1-d input
            mean_func = pm.gp.mean.Constant(mu)

        with lgcp_model:
            gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
            f = gp.prior("f", X=centroids)

            # Likelihood
            y = pm.Poisson("y", mu=pm.math.exp(f), observed=cell_counts)
            log_intensity = gp.prior("log_intensity", X=centroids)
            intensity = pm.math.exp(log_intensity)

            rates = intensity * area_per_cell
            counts = pm.Poisson("counts", mu=rates, observed=cell_counts)

        x_new = np.linspace(boundary_lower, boundary_upper, 70)
        x_new = np.expand_dims(x_new, 1)
        if run_new_samples:
            with lgcp_model:
                trace = pm.sample(1000, tune=2000, target_accept=0.95)

            az.summary(trace, var_names=["mu", "rho", "variance"])

            with lgcp_model:
                intensity_new = gp.conditional("log_intensity_new", Xnew=x_new)
                app_trace = pm.sample_posterior_predictive(
                    trace, var_names=["log_intensity_new"]
                )

            trace.extend(app_trace)
            intensity_samples = np.exp(
                trace.posterior_predictive["log_intensity_new"]
            )
            np.save(
                "intensity_samples_mcmc_1d_synth_{}.npy".format(i),
                intensity_samples,
            )
            np.save(
                "intensity_samples_mcmc_1d_synth_{}_inputs.npy".format(i),
                x_new,
            )
        else:
            intensity_samples = np.load(
                "intensity_samples_mcmc_1d_synth_{}.npy".format(i)
            )

        fig = plt.figure(figsize=(5, 4))

        mean = intensity_samples.mean(axis=0).mean(axis=0)
        # breakpoint()
        plt.plot(
            x_new,
            mean,
        )

        plt.title("$E[\\lambda(s) \\vert Y]$")
        # plt.colorbar(label="Posterior mean")
        # plt.scatter(data["x"], data["y"], s=6, color="k")
        plt.show()
