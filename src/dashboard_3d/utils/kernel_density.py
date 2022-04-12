"""Module comprising of statsmodels KDE model API overrides.

The KDEMultivariate class in this module was taken verbatim with one modification from statsmodels commit hash ce02f8e8ade54, see [0].
We have modified the KDEMultivariate class to allow for the 'max_nbytes' parameter to be passed to the _compute_efficient() function.
This was to fix an issue when n_jobs=-1 to run in parallel mode, where fitting would break for large datasets, for full details please see [1].

[0] https://github.com/statsmodels/statsmodels/pull/6058/commits/ce02f8e8ade546fb0e8a1b155add8aae7362242f
"""
import copy

import joblib
import numpy as np
from statsmodels.nonparametric._kernel_base import (EstimatorSettings,
                                                    _adjust_shape,
                                                    _compute_subset,
                                                    _get_type_pos, has_joblib)
from statsmodels.nonparametric.kernel_density import \
    KDEMultivariate as _KDEMultivariate


class KDEMultivariate(_KDEMultivariate):
    """Multivariate kernel density estimator.

    Modified with an additional parameter for max_nbytes and an override of the _compute_efficient function to pass an
    extra parameter to joblib.
    """

    def __init__(self, data, var_type, bw=None, defaults=None, max_nbytes="1M"):
        """Initialisation."""
        self.var_type = var_type
        self.k_vars = len(self.var_type)
        self.data = _adjust_shape(data, self.k_vars)
        self.data_type = var_type
        self.nobs, self.k_vars = np.shape(self.data)
        # Addition of a new parameter for control over max_nbytes
        self.max_nbytes = max_nbytes
        if self.nobs <= self.k_vars:
            raise ValueError(
                "The number of observations must be larger than the number of variables."
            )
        defaults = EstimatorSettings() if defaults is None else defaults
        self._set_defaults(defaults)
        if not self.efficient:
            self.bw = self._compute_bw(bw)
        else:
            self.bw = self._compute_efficient(bw)

    def _compute_efficient(self, bw):
        """Computes bandwidth.

        Function override to pass an extra parameter 'max_nbytes' to joblib call.
        """
        if bw is None:
            self._bw_method = "normal_reference"
        if isinstance(bw, str):
            self._bw_method = bw
        else:
            self._bw_method = "user-specified"
            return bw

        nobs = self.nobs
        n_sub = self.n_sub
        data = copy.deepcopy(self.data)
        n_cvars = self.data_type.count("c")
        co = 4  # 2*order of continuous kernel
        do = 4  # 2*order of discrete kernel
        _, ix_ord, ix_unord = _get_type_pos(self.data_type)

        # Define bounds for slicing the data
        if self.randomize:
            # randomize chooses blocks of size n_sub, independent of nobs
            bounds = [None] * self.n_res
        else:
            bounds = [(i * n_sub, (i + 1) * n_sub) for i in range(nobs // n_sub)]
            if nobs % n_sub > 0:
                bounds.append((nobs - nobs % n_sub, nobs))

        n_blocks = self.n_res if self.randomize else len(bounds)
        sample_scale = np.empty((n_blocks, self.k_vars))
        only_bw = np.empty((n_blocks, self.k_vars))

        class_type, class_vars = self._get_class_vars_type()
        if has_joblib:
            # `res` is a list of tuples (sample_scale_sub, bw_sub)
            # max_nbytes parameter has been added to allow for control over memory mapping
            res = joblib.Parallel(n_jobs=self.n_jobs, max_nbytes=self.max_nbytes)(
                joblib.delayed(_compute_subset)(
                    class_type,
                    data,
                    bw,
                    co,
                    do,
                    n_cvars,
                    ix_ord,
                    ix_unord,
                    n_sub,
                    class_vars,
                    self.randomize,
                    bounds[i],
                )
                for i in range(n_blocks)
            )
        else:
            res = []
            for i in range(n_blocks):
                res.append(
                    _compute_subset(
                        class_type,
                        data,
                        bw,
                        co,
                        do,
                        n_cvars,
                        ix_ord,
                        ix_unord,
                        n_sub,
                        class_vars,
                        self.randomize,
                        bounds[i],
                    )
                )

        for i in range(n_blocks):
            sample_scale[i, :] = res[i][0]
            only_bw[i, :] = res[i][1]

        s = self._compute_dispersion(data)
        order_func = np.median if self.return_median else np.mean
        m_scale = order_func(sample_scale, axis=0)
        # TODO: Check if 1/5 is correct in line below!
        bw = m_scale * s * nobs ** (-1.0 / (n_cvars + co))
        bw[ix_ord] = m_scale[ix_ord] * nobs ** (-2.0 / (n_cvars + do))
        bw[ix_unord] = m_scale[ix_unord] * nobs ** (-2.0 / (n_cvars + do))

        if self.return_only_bw:
            bw = np.median(only_bw, axis=0)

        return bw
