"""
The log likelihood of the model's fit to a noisy disc throw.
"""
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from frispy import Disc


class SingleThrowExperiment:
    def __init__(self, fpath: str, clockwise: bool = True):
        self.df = pd.read_pickle(fpath)
        assert len(self.df) > 1  # need more than one data point
        # Create a frozen prior on the initial velocities
        self.v_prior = self.get_initial_velocity_distribution()
        self.disc = Disc()
        # Prior on the angles is an isotropic normal distribution centered at 0
        # with a standard deviaton of a half a radian (28 degrees)
        self.angle_prior = multivariate_normal(mean=[0, 0], cov = .25)
        # Prior on phi and theta angular velocities
        self.pt_prior = multivariate_normal(mean=[0, 0], cov=0.0025)
        # Prior on dgamma
        # 60 radians/s is about 10 rotations/sec
        # the prior is very wide (50%)
        self.dgamma_prior = norm(loc=np.log(60), scale=0.5)

    def get_initial_velocity_distribution(self):
        noise = self.df["errorbar"]
        dt = self.df["times"].diff()[
            1
        ]  # the difference between the first and second time
        # Variance estimate is the quadrature sum of the noise on the first two points
        cov = np.eye(3) * (noise.iloc[1] ** 2 + noise.iloc[0] ** 2) / dt ** 2

        # The mean estimate is the finite difference between the first two measurements
        mean = np.zeros(3)
        for i, key in enumerate(list("xyz")):
            mean[i] = self.df[key].diff()[1] / dt
        return multivariate_normal(mean=mean, cov=cov)

    def sample_initial_velocities(self, n_samples: int) -> np.ndarray:
        return self.v_prior.rvs(size=n_samples)

    def unpack_params(self, params):
        return {"v": params[:3], "angles": params[3:6], "dphi_dtheta": params[6:8], "ln_dgamma": params[8], "model_params": params[9:]}

    def ln_priors(self, params) -> float:
        dct = self.unpack_params(params)

        # Priors
        v_prior = self.v_prior.logpdf(dct["v"])
        angle_prior = self.angle_prior.logpdf(dct["angles"])
        av_prior = self.pt_prior.logpdf(dct["dphi_dtheta"])
        dg_prior = self.dgamma_prior.logpdf(dct["ln_dgamma"])
        return v_prior + angle_prior + av_prior + dg_prior

    def ln_likelihood(self, params) -> float:
        dct = self.unpack_params(params)
        # Update the disc's initial conditions

        # Update the model
        pass