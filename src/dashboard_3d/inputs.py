"""Manage data and model inputs for the application."""
import numpy as np

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
        self.numpoints = config["numpoints"]

        # data
        self.df = None
        self.tags = None

        # models
        self.kde = None
        self.pdf_x = None
        self.pdf_y = None
        self.pdf_z = None
        self.pdf_values = None
        self.grid_coords = None

        self.get_data()
        self.fit_models()

    def get_data(self):
        """Generate mixture data."""
        self.df, self.tags = generate_mixture_data(self.n_samples, self.mixture_params)

    def fit_models(self):
        """fit statistical models."""

        # fit health model
        kwargs = dict(area_validation=False, numPoints=self.numpoints)
        if self.model_type == "MLCV":
            self.kde = HealthModelMLCV(**kwargs)
        elif self.model_type == "BP11":
            self.kde = HealthModelBP11(**kwargs)

        self.kde.fit(self.df, taglist=self.tags, tag_descriptions=None)

        # TODO: Fix marginalising for MLCV (only work for BP11)
        # marginalise:
        # integral: z grid points dy, y grid points dx
        self.pdf_x = simps(
            simps(self.kde.pdf, x=self.kde.axes[2], axis=1), x=self.kde.axes[1], axis=0
        )
        # integral: x grid points dz, z grid points dx
        self.pdf_y = simps(
            simps(self.kde.pdf, x=self.kde.axes[0], axis=2), x=self.kde.axes[2], axis=0
        )
        # integral: y grid points dz, x grid points dy
        self.pdf_z = simps(
            simps(self.kde.pdf, x=self.kde.axes[1], axis=2), x=self.kde.axes[0], axis=1
        )

        # probability density grid coordinates and model output densities
        self.pdf_values = self.kde.pdf_masked.data.flatten()

        gridpoints = np.array(np.meshgrid(*self.kde.axes, indexing="ij"))
        dims = [x.shape[0] for x in self.kde.axes]
        self.grid_coords = gridpoints.T.reshape([np.prod(dims), len(dims)])

        #### UNIT TESTS
        # gridpoints = np.array(np.meshgrid(*self.kde.axes, indexing='ij'))
        # dims = [x.shape[0] for x in self.kde.axes]
        # grid_coords = gridpoints.T.reshape([np.prod(dims), len(dims)])
        # # grid_coords, _, _, _ = generate_grid(kde.axes, len(kde.axes[0]))
        # actual =self.kde.pdf_masked.data.flatten()
        # desired, _ = self.kde.evaluate(grid_coords)
        # int_e = np.sum((actual.data - desired) ** 2) * np.prod(self.kde.deltaX)
        # print(f"\tI.S.E between fit and inference = {int_e}")
        # assert np.isclose(actual, desired).all()
        #
        # gridpoints = np.array(np.meshgrid(*self.kde.axes, indexing='xy'))
        # dims = [x.shape[0] for x in self.kde.axes]
        # grid_coords = gridpoints.T.reshape([np.prod(dims), len(dims)], order='F')
        # # grid_coords, _, _, _ = generate_grid(kde.axes, len(kde.axes[0]))
        # center = int(len(grid_coords) / 2) - 3
        # actual = self.kde.pdf_masked.data.flatten(order='F')[center]
        # desired, _ = self.kde.evaluate([grid_coords[center]])
        # print(f'{actual} : {float(desired)}')
        # assert np.isclose(actual, desired)

        return self
