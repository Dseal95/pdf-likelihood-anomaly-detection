"""Module comprising process health KDE model API.

TODO - add some configurable error handling for failed MLCV fits (when pdfobj and bw are nan)
"""


import copy
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from fastkde import fastKDE
except ImportError:
    pass

from scipy.integrate import simps
from statsmodels.nonparametric.kernel_density import EstimatorSettings

from utils.kernel_density import KDEMultivariate
from utils.simulation import generate_grid


class HealthModelGeneric:
    """Generic class encapsulates nD health models in a sci-kit style API.

    Different KDE algorithms are accommodated by overriding
    `~health_models.HealthModelGeneric.fit` and `~health_models.HealthModelGeneric.fit`

    Instance Attributes:
        pdf_masked (maskedarray):
            The fitted nD capability envelope [numPoints x numPoints...]

        axes (list of arrays):
            Uniformly spaced grid points of the capability envelope, one array per axis

        cpr (list of arrays):
            Likelihood limit lookup table (L(x), Pr)

        deltaX (list of floats):
            Spacing of each grid axes

        taglist (list):
            Health measures the model is trained on

        data (DataFrame):
            Model training data

        context (dict):
            Context parameters used to subset the training data

        pdfobj (object):
            The underlying KDE model instance

    Class Attributes:
        alpha (float):
            Default probability threshold (1-confidence level)
    """
    # TODO: add non-deterministic and sample quantile integration options
    # TODO: add a predict method for returning status for a new point, given alpha
    # TODO: add method for computing marginals

    alpha = 0.05

    def __init__(self):
        """Initialisation."""
        self.data = {}
        self.taglist = []
        self.pdf = np.empty(0)
        self.pdf_masked = np.ma.empty(0)
        self.context_id = None
        self.trained = False
        self.cpr = None
        self.logger = logging.getLogger(__name__)
        self.pdfobj = None
        self.data_lims = []
        self.deltaX = None
        self.axes = None
        self.kdekwargs = None
        self.model_kwargs = None
        self.numPoints = None
        self.area_validation = True
        self.tag_descriptions = {}
        self.grid_coords = None

        # self.deltaP = None


    @property
    def name(self):
        """Return name of Health Unit."""
        return "_".join(self.taglist)

    @property
    def area(self):
        """Return area under the PDF calculated by rectangular numerical integration."""
        if self.trained:
            return np.sum(self.pdf * np.prod(self.deltaX))
        else:
            self.logger.warning('PDF area cannot be calculated (model not trained)')
            return None

    def _reset(self):
        """Reset the instance to default values."""
        self.__init__(**self.model_kwargs)

    def fit(self):
        """Fit an nD KDE model and return result on the algorithm's calculated uniform grid.

        Args:
            df (dataframe):
                training data
            taglist (list of strings):
                names of the tags to include in the model
            cxt_kwargs (dict):
                context_id definition of training data

        Returns:
            pdf_masked (maskedarray):
                the estimated pdf masked to raw data domain
            axes (list of arrays)
                individual axes of the uniform modelling grid
        """
        return NotImplementedError

    def evaluate(self):
        """Return the likelihood for new observations on a trained nD model.

        For a fixed model, f(x|theta) = L(x)
        i.e. the density at a particular value of x, given that the model is true

        Args:
            xpoints (array_like):
                Points at which the model should be queried

                Accepts an ndarray or list of tuples of shape [numDataPoints, numVariables],
                where column order is assumed to be as per `self.taglist`

                If a Dataframe is provided then the xpoints will be extracted using
                the names of the variables the model is trained for.

        Returns:
            pdf_points (array):
                rank 2 array of the Likelihood at each evaluation point

            x_predict (array):
                rank 2 array of the evaluation points
        """
        return NotImplementedError

    def score(self, x=None, ll=None):
        """Return the health score at x.

        i.e.
            Pr[L(x) <= l] | {theta,C}

        The probability of a likelihood less than or equal to the likelihood at x

        Args:
            x: Union[pd.Dataframe, np.array] (optional)
                Data points at which to compute the Pr, df or array_like
            ll: np.array (optional)
                Likelihoods at which to compute the Pr, array_like

        Returns:
            ll : the likelihood at x
            score: the health score at x

        # TODO: ensure output format is consistent for single and list inputs
        """
        if not self.trained:
            raise RuntimeError('Model has not been trained')

        # if observations are provided, first evaluate their likelihoods
        self.logger.info('Calculating health score(s)...')
        if x is not None:
            if ll is not None:
                self.logger.warning(
                    'Overwriting likelihood input using point predictions')
            ll, _ = self.evaluate(x)

        # for each likelihood, perform piecewise linear interpolation
        # of the cumulative Pr function
        score = np.interp(ll, *self.cpr)

        return score, ll

    def save(self, model_dir):
        """Dumps the model instance to a local .pkl file.

        Note: File names are sanitised to avoid the nasty characters appearing in
        some tag names
        """
        if self.trained:

            model_name = str().join(self.taglist)
            model_name = model_name.replace("\\", "_").replace("/", "_").replace(":", "_")
            model_path = Path(model_dir) / (model_name + '.pkl')

            # create the directory
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # serialise
            with open(model_path, 'wb') as mfile:
                pickle.dump(self, mfile)

            self.logger.info(f'Model has been serialised @ {model_path}')

        else:
            self.logger.error('Model is not trained, export cancelled.')

    @classmethod
    def load(cls, model_path):
        """Load the Health Unit from model path."""
        with open(model_path, 'rb') as pickle_file:
            content = pickle.load(pickle_file)

        return content

    @staticmethod
    def pr_interval(pdf, alpha=None):
        """Return the cumulative probability function.

        i.e. Pr(l):

            Pr[L(x) <= l] | {theta,C}

        If a probability threshold is specified (alpha),
        function also returns the likelihood threshold, l_thresh, where:

            Pr[L(x) > l_thresh] | {theta,C} = 1 - alpha

        Method estimates the area of the region R(f) an
        nD p(x) distribution as a function of l

        Args:
            pdf : np.array
                the nD pdf to be integrated
            alpha : float (optional)
                probability threshold

        Returns:
            lkld_x : likelihood values of the Pr function
            cumpr_y :
        """
        # this is a deterministic 'approximation' to a naive Monte Carlo method,
        # i.e. integration on a very high density uniform grid
        cumpr_y = np.cumsum(np.sort(pdf.flatten())) / np.sum(pdf.flatten())
        lklhd_x = np.sort(pdf.flatten())

        # compute the likelihood threshold which yields a cumulative probability
        # Pr[L(x) <= l] = alpha, equivalently
        # Pr[X e R(f_alpha)] > 1-alpha; where R(falpha) = {x: L(x) > l}

        if alpha is not None:
            lklhd_thresh = lklhd_x[cumpr_y >= alpha][0]
            return lklhd_x, cumpr_y, lklhd_thresh
        else:
            return lklhd_x, cumpr_y

    def _ndmask(self):
        """Generate a masked version of an nD pdf."""
        # mask nD result to the raw data domain for plotting functions
        # find limits in each axis (+hack for duplicate column entries)
        self.data_lims = []
        for tag in self.taglist:
            data = self.data.loc[:, ~self.data.columns.duplicated()]
            self.data_lims.append((np.min(data[tag]), np.max(data[tag])))

        # create separate axis masks
        if type(self.axes) == list:
            axes = self.axes
        elif self.axes.ndim < 2:
            axes = self.axes[np.newaxis, :]
        masks = []
        for ax, lim in zip(axes, self.data_lims):
            masks.append(~((ax >= lim[0]) & (ax <= lim[1])))

        # create an nD grid of masks
        mask_meshgrid = np.meshgrid(*masks)

        # create a full mask
        mask_full = mask_meshgrid[0]
        for mask in mask_meshgrid:
            mask_full = mask_full | mask

        return np.ma.masked_array(self.pdf, mask_full)

    def _area_test(self):
        """Test the area under a trained pdf.

        Performs deterministic multiple integration over the trained non-masked pdf
        """
        area = []
        # integration by Simpson
        cumpr = self.pdf.copy()
        for ax in self.deltaX:
            cumpr = simps(y=cumpr, dx=ax, axis=0)
        area.append(cumpr)

        # integration by rectangles
        area.append(np.sum(self.pdf * np.prod(self.deltaX)))

        # test whether integrated area is ~= 1.0
        test_result = np.isclose(1, area[1], atol=1e-2)

        if test_result:
            self.logger.info(f'\tIntegrated area under the pdf (Simpson, rectangles) = ('
                             f'{area[0]},{area[1]})')
        else:
            self.logger.warning(f'\tIntegrated area under the pdf (Simpson, rectangles)'
                                f' = ({area[0]},{area[1]})')

        # only assert area requirement if area_validation is set to True
        if self.area_validation:
            assert test_result, 'KDE area =! 1.0'

        return test_result

    def _parsepoints(self, xpoints):
        """Convert input points into a rank 2 array that is transposed with respect to input data.

        Args:
            xpoints : array_like, DataFrame
                the data to parse

        Returns:
            x_predict (array):
                points at which to evaluate the data of shape [numVariables,numDataPoints]
                this is transposed with respect to the input

        TODO: test that parsing is handling arrays properly in 3D
        """
        # if no input points are specified, perform prediction on the training data
        if xpoints is None:
            xpoints = self.data

        # if DataFrame, directly subset to the trained tags
        if isinstance(xpoints, type(pd.DataFrame())):
            df_x = xpoints[self.taglist]

        # if a list or array, convert to a DataFrame
        elif isinstance(xpoints, (list, np.ndarray)):
            df_x = pd.DataFrame(xpoints, columns=self.taglist)
            self.logger.info(f'\tEvaluation points converted to dataframe '
                             f'assuming array column order is {self.taglist}')
        else:
            msg = 'Incorrect data input type. Expected DataFrame or array-like'
            raise(TypeError, msg)

        # convert to an array
        x_predict = np.array(df_x.values, copy=True, dtype=np.float).T

        # check the rank of the input data points
        # if the data are an array, promote the data to a rank-1 array with only 1 column
        dataRank = len(np.shape(x_predict))
        if (dataRank == 1):
            x_predict = np.array(x_predict[np.newaxis, :], dtype=np.float)
        if (dataRank > 2):
            raise ValueError("cannot broadcast x_predict to a rank-2 array "
                             "of shape [numDataPoints,numVariables]")

        # check that shape is correct
        if x_predict.T.shape[1] != len(self.taglist):
            msg = f"Input data format is incorrect. Expected: " \
                f"[numDataPoints, numVariables={len(self.taglist)}], " \
                f"Actual: {x_predict.T.shape}"
            raise ValueError(msg)

        return x_predict


class HealthModelBP11(HealthModelGeneric):
    """Subclass performs KDE using the BP11 Kernel.

    BP11 uses an objective, data driven kernel that does not rely on a prior.
    For more info see Bernacchia and Pigolotti, 2011

    fastKDE uses a non-uniform FFT to perform density estimation using the
    BP11 'self-consistent' kernel.

    Parent Instance Attributes (fastKDE):
        axes (list of arrays):
            Uniformly spaced grid points of the capability envelope, one array per axis

        deltaX (list of floats):
            Spacing of each grid axes

        deltaP (float):
            Vertical correction of the modelled density function
    """
    # TODO: add non-deterministic and sample quantile integration options
    # TODO: add a predict method for returning class for a new point, given alpha
    # TODO: add method for computing marginals

    defaultconfig = {'doApproximateECF': True,
                     'ecfPrecision': 1,
                     'numPointsPerSigma': 50,
                     'numPoints': 1025,
                     'doSaveMarginals': False,
                     'positiveShift': True
                     }

    def __init__(self, **model_kwargs):
        """Initialisation."""
        # instantiate generic base class only
        super().__init__()

        # override default kde config with any inputs provided by user
        self.kdekwargs = copy.deepcopy(self.defaultconfig)
        for key, val in model_kwargs.items():
            self.kdekwargs[key] = val

        # initialise instance attributes
        self.numPoints = copy.deepcopy(self.kdekwargs['numPoints'])
        self.area_validation = copy.deepcopy(
            self.kdekwargs.setdefault('area_validation', self.area_validation))

        # do not initialise parent instance attributes
        self.deltaP = 0.0
        # self.deltaX = None
        # self.axes = None

        # save input for stateful reset
        self.model_kwargs = model_kwargs

        # strip the kwargs needed only for statsmodels kde API
        self.kdekwargs.pop('area_validation')

    def fit(self, df, taglist, tag_descriptions, cxt_kwargs=None):
        """Fit an nD KDE model and return result on a uniform grid.

        Overrides parent class method
        """
        # input conversion
        if type(taglist) is str:
            taglist = [taglist]

        # reset the instance
        self._reset()
        self.taglist = taglist

        # store the training context_id
        self.context_id = cxt_kwargs

        self.tag_descriptions = tag_descriptions

        # subset data and drop missing values
        self.data = df[taglist].dropna()

        # perform fastkde estimation on a fixed grid (also establishes correction shift)
        self.logger.info(
            f'Training KDE model: estimating pdf p(X) via fastKDE algorithm; '
            f'X = {self.taglist}...')
        # this call overrides two common attributes from the generic class (axes, deltaX)
        self.pdfobj = fastKDE.fastKDE(data=self.data.values.T, **self.kdekwargs)
        # super().__init__(data=self.data.values.T, axes=None, **self.kdekwargs)
        self.trained = True
        self.pdf = self.pdfobj.pdf
        self.axes = self.pdfobj.axes
        self.deltaX = self.pdfobj.deltaX
        # self.deltaP = self.pdfobj.positiveShift

        # integrate pdf to check area = 1
        self._area_test()

        # mask the pdf to the domain
        self.pdf_masked = self._ndmask()

        # calculate the cumulative probability function over the whole (unmasked) domain
        self.cpr = self.pr_interval(self.pdf)

        return self.pdf_masked, self.axes

    def evaluate(self, xpoints=None):
        """Return the likelihood for new observations on a trained nD model.

        Overrides parent class method
        """
        # TODO: add assertion on inference points outside the raw data domain

        # if the model has not been trained
        if not self.trained:
            raise RuntimeError('Model has not yet been trained. Provide input data.')

        # clean up the input points
        x_predict = self._parsepoints(xpoints)

        # perform inference
        self.logger.info(f'Estimating the pdf p(X) at n={x_predict.shape[1]} points;'
                         f' X = {self.taglist}...')
        dftkwargs = self.kdekwargs.copy()
        dftkwargs['doFFT'] = False
        # calculate the PDF in Fourier space
        self.pdfobj = fastKDE.fastKDE(data=self.data.values.T, **dftkwargs)

        # complete the Fourier-space calculation of the PDF
        self.pdfobj.applyBernacchiaFilter()

        # calculate the PDF at the requested points
        pdf_points = self.pdfobj.__transformphiSC_points__(x_predict)

        # # apply the trained pdf area correction
        # if self.deltaP is not None:
        #     pdf_points -= self.deltaP

        #     # coerce remaining negative points to zero (roundoff errors)
        #     pdf_points[pdf_points < 0.0] = 0.0

        return pdf_points, x_predict


class HealthModelMLCV(HealthModelGeneric):
    """Subclass performs KDE using a Gaussian kernel with bandwidths selected via ML-CV)."""
    defaultconfig = {'efficient': True,
                     'randomize': True,
                     'n_res': 32,
                     'n_sub': 256,
                     'n_jobs': -1,
                     'numPoints': 256,
                     'max_nbytes': '1M'}
    """
    Copied from `statsmodels.EstimatorSettings` (for kwargs used in the defaultconfig)
    ------------------------------------------------------------------------
    efficient : bool, optional
        If True, the bandwidth estimation is to be performed
        efficiently -- by taking smaller sub-samples and estimating
        the scaling factor of each subsample.  This is useful for large
        samples (nobs >> 300) and/or multiple variables (k_vars > 3).
        If False (default), all data is used at the same time.
    randomize : bool, optional
        If True, the bandwidth estimation is to be performed by
        taking `n_res` random resamples (with replacement) of size `n_sub` from
        the full sample.  If set to False (default), the estimation is
        performed by slicing the full sample in sub-samples of size `n_sub` so
        that all samples are used once.
    n_sub : int, optional
        Size of the sub-samples.  Default is 50.
    n_res : int, optional
        The number of random re-samples used to estimate the bandwidth.
        Only has an effect if ``randomize == True``.  Default value is 25.
    """

    def __init__(self, **model_kwargs):
        """Initialisation."""
        # instantiate base class
        super().__init__()

        # override default kwargs with any provided inputs
        self.kdekwargs = copy.deepcopy(self.defaultconfig)
        for key, val in model_kwargs.items():
            self.kdekwargs[key] = val

        # initialise instance attributes
        self.numPoints = copy.deepcopy(self.kdekwargs['numPoints'])
        self.area_validation = copy.deepcopy(
            self.kdekwargs.setdefault('area_validation', self.area_validation))

        # assign inputs for stateful reset
        self.model_kwargs = model_kwargs

        # set the max_nbytes parameter for joblib
        self.max_nbytes = copy.deepcopy(self.kdekwargs['max_nbytes'])

        # strip the kwargs needed only for statsmodels kde API
        self.kdekwargs.pop('max_nbytes')
        self.kdekwargs.pop('numPoints')
        self.kdekwargs.pop('area_validation')

    def fit(self, df, taglist, tag_descriptions, cxt_kwargs=None):
        """Fit an nD KDE model and return result on a uniform grid.

        Overrides parent class method

        This is broken into two distinct steps:
            1) Gaussian kernel bandwidth selection via MLCV
            2) Evaluation of density function on a uniform grid
        """
        # input conversion
        if type(taglist) is str:
            taglist = [taglist]

        # reset the instance
        self._reset()
        self.taglist = taglist

        # store the training context_id
        self.context_id = cxt_kwargs

        self.tag_descriptions = tag_descriptions

        # subset data and drop missing values
        self.data = df[taglist].dropna()

        # select bandwidths via MLCV
        self.logger.info(
            f'Training KDE model step 1/2: selecting kernel bandwidth via MLCV; '
            f'X = {self.taglist}...')

        # create estimator settings object
        est_config = EstimatorSettings(**self.kdekwargs)

        # generate the grid
        griditems = generate_grid(self.data.values, self.numPoints)
        self.grid_coords, _, dims, self.axes = griditems
        self.deltaX = [np.diff(x[0:2])[0] for x in self.axes]

        # perform bandwidth selection
        vartype = str().join(['c'] * self.data.values.shape[1])
        self.pdfobj = KDEMultivariate(self.data.values,
                                      var_type=vartype,
                                      bw='cv_ml',
                                      defaults=est_config,
                                      max_nbytes=self.max_nbytes)

        self.logger.info(
            f'\tMLCV selected bandwidths: {list(self.pdfobj.bw)}')

        self.trained = True

        self.logger.info(
            f'Training KDE model step 2/2: estimating pdf p(X) on uniform grid: {dims}...')

        # use selected bandwidth to estimate the PDF on the internal model grid
        pdf_pairs, _ = self.evaluate(self.grid_coords)

        self.pdf = np.reshape(pdf_pairs, *[dims])

        # integrate pdf to check area = 1
        self._area_test()

        # mask the pdf to the domain
        self.pdf_masked = self._ndmask()

        # calculate the cumulative probability function over the whole (unmasked) domain
        self.cpr = self.pr_interval(self.pdf)

        return self.pdf_masked, self.axes

    def evaluate(self, xpoints=None):
        """Return the likelihood for new observations on a trained nD model.

        Overrides parent class method
        """
        # if the model has not been trained
        if not self.trained:
            raise RuntimeError('Model has not yet been trained. Provide input data.')

        # clean up the input points
        x_predict = self._parsepoints(xpoints).T

        # perform inference
        self.logger.info(f'Estimating the pdf p(X) at n={x_predict.shape[0]} points;'
                         f' X = {self.taglist}...')

        # calculate the PDF at the requested points
        pdf_points = self.pdfobj.pdf(x_predict).tolist()

        return pdf_points, x_predict
