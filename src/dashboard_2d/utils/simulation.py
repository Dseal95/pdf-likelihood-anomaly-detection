"""Module comprising reusable simulation functionality."""

from typing import List

import numpy as np
import pandas as pd


def generate_3D_mixture(n_samples: int, mixture_params: List[list]):
    """Generate synthetic 3D Gaussian mixture data.

    Args:
    n_samples: Number of samples to generate
    mixture_params: Parameters of each component in the mixture
        Each component in the mixture is defined by a list of its parameters,
        [mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, rho_xy, rho_xz, rho_yz].

    Returns:
        (tuple): tuple containing:

            df (pd.Dataframe): dataframe containing synthetic data
            tags (list): tag names
    """

    def build_cov_mat(
        sx: float, sy: float, sz: float, rho_xy: float, rho_xz: float, rho_yz: float
    ):
        """Assemble the covariance matrix."""
        c_xy = rho_xy * np.sqrt(sx * sy)
        c_xz = rho_xz * np.sqrt(sx * sz)
        c_yz = rho_yz * np.sqrt(sy * sz)
        return [[sx**2, c_xy, c_xz], [c_xy, sy**2, c_yz], [c_xz, c_yz, sz**2]]

    # number of mixtures
    n = int(n_samples / len(mixture_params))

    # generate samples
    data = []
    for dist in mixture_params:
        mu = dist[:3]
        gCovMat = build_cov_mat(*tuple(dist[3:]))
        data.append(np.random.multivariate_normal(mu, gCovMat, n).T)
    data = np.concatenate(tuple(data), axis=1)

    # outputs
    tags = ["GMM_x", "GMM_y", "GMM_z"]
    df = pd.DataFrame(data.T, columns=tags)

    return df, tags


def generate_2D_mixture(n_samples: int, mixture_params: List[list]):
    """Generate synthetic 2D Gaussian mixture data.

    Args:
    n_samples: Number of samples to generate
    mixture_params: Parameters of each component in the mixture
        Each component in the mixture is defined by a list of its parameters,
        [mu_x, mu_y, sigma_x, sigma_y, rho_xy].

    Returns:
        (tuple): tuple containing:

            df (pd.Dataframe): dataframe containing synthetic data
            tags (list): tag names
    """

    def build_cov_mat(sx: float, sy: float, rho: float):
        """Assemble the covariance matrix."""
        c_xy = rho * np.sqrt(sx * sy)
        return [[sx**2, c_xy], [c_xy, sy**2]]

    # number of mixtures
    n = int(n_samples / len(mixture_params))

    # generate samples
    data = []
    for dist in mixture_params:
        mu = dist[:2]
        gCovMat = build_cov_mat(*tuple(dist[2:]))
        data.append(np.random.multivariate_normal(mu, gCovMat, n).T)
    data = np.concatenate(tuple(data), axis=1)

    # outputs
    tags = ["GMM_x", "GMM_y"]
    df = pd.DataFrame(data.T, columns=tags)

    return df, tags


def generate_grid(xdata, n_grid=256):
    """Create uniformly spaced grid from nD axis vectors.

    Args:
        xdata: array_like
            raw data for each variable
            if list, then of length [numVariables]
            if array, then of shape [numDatapoints, numVariables]

        n_grid: int or list of int
            number of grid points to generate across the domain of each variable.
            If int, the grid is equally sized in each dimension

    TODO: clean up this ridiculous parsing
    """
    # parse inputs
    # convert xdata to array
    if not isinstance(xdata, np.ndarray):
        xdata = np.array(xdata).T
    # convert n_grid to list and check shape
    if not isinstance(type(n_grid), list):
        n_grid = [n_grid] * xdata.shape[1]
    else:
        assert xdata.shape[1] == len(n_grid), "Grid inputs not compatible"

    # coerce
    n_grid = [int(x) for x in n_grid]

    # create uniformly spaced grid axes
    axes = []
    for ax, n in zip(xdata.T, n_grid):
        axes.append(np.linspace(ax.min(), ax.max(), num=n))

    # create a grid array of size (n x dims[0] x dims[1]...)
    grid = np.array(np.meshgrid(*axes))

    # calculate the grid dimensions
    dims = [x.shape[0] for x in axes]

    # generate pairs of all the grid points
    grid_coords = grid.T.reshape([np.prod(dims), len(dims)], order="F")

    return grid_coords, grid, dims, axes
