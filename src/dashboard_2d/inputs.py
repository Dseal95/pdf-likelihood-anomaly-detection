"""Manage data and model inputs for the application."""

import numpy as np
import pandas as pd
from scipy.integrate import simps

from utils.models import HealthModelBP11, HealthModelMLCV
from utils.simulation import generate_mixture_data


class Inputs:
    """Container class for synthetic data and models."""

    def __init__(self, config: dict):
        """Generate data and fit models based on config."""
        # parameters
        self.n_samples = config["n_samples"]
        self.mixture_params = config["mixture_params"]
        self.model_type = config["model_type"]
        self.num_points = config["num_points"]

        # data
        self.df = None
        self.tags = None

        # models
        self.kde = None
        self.pdf_x = None
        self.pdf_y = None

        self.get_data()
        self.fit_models()

    def get_data(self):
        """Generate mixture data."""
        self.df, self.tags = generate_mixture_data(
            n_samples=self.n_samples, mixture_params=self.mixture_params
        )

    def fit_models(self):
        """fit statistical models."""

        # fit health model
        kwargs = dict(area_validation=False, num_points=self.num_points)
        if self.model_type == "MLCV":
            self.kde = HealthModelMLCV(**kwargs)
        elif self.model_type == "BP11":
            self.kde = HealthModelBP11(**kwargs)
        self.kde.fit(df=self.df, taglist=self.tags, tag_descriptions=None)

        # marginalise
        self.pdf_y = simps(y=self.kde.pdf, x=self.kde.axes[0], axis=1)
        self.pdf_x = simps(y=self.kde.pdf, x=self.kde.axes[1], axis=0)

        return self
