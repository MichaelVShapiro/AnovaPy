from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma, hyp1f1
from scipy.stats import chi2, kurtosis, lognorm, norm, shapiro, skew

__all__ = ["doornik", "energy", "henze", "mardia", "royston", "shapiro_francia"]

HenzeResult = namedtuple("HenzeResult", ["statistic", "pvalue"])
DoornikResult = namedtuple("DoornikResult", ["statistic", "pvalue"])
EnergyResult = namedtuple("EnergyResult", ["statistic", "pvalue"])
RoystonResult = namedtuple("RoystonResult", ["statistic", "pvalue"])
ShapiroFranciaResult = namedtuple("ShapiroFranciaResult", ["statistic", "pvalue"])


@dataclass
class MardiaResult:
    """
    Result of `anovapy.normal.mardia`.

    The test decomposes into two independent components:
    a skewness test and a kurtosis test. Data is consistent with
    multivariate normality if both p-values exceed the chosen
    significance level.

    Attributes
    ----------
    skew : float
        The sample multivariate skewness. This really is just a descriptive statistic quantifying the degree of asymmetry; equals 0 under multivariate normality.
    skew_statistic : float
        The skew test statistic
    skew_pvalue : float
        Computed p-value of skewness
    kurtosis : float
        The computed multivariate kurtosis. This is a descriptive statistic quantifying tail heaviness relative to a multivariate normal; equals 0 under multivariate normality.
    kurtosis_statistic : float
        The kurtosis test statistic
    kurtosis_pvalue : float
        Computed p-value of kurtosis
    """

    skew: float
    skew_statistic: float
    skew_pvalue: float
    kurtosis: float
    kurtosis_statistic: float
    kurtosis_pvalue: float

    def __repr__(self):
        return (
            f"MardiaResult("
            f"  skew={self.skew}, "
            f"  skew_statistic={self.skew_statistic}, "
            f"  skew_pvalue={self.skew_pvalue}, "
            f"  kurtosis={self.kurtosis}, "
            f"  kurtosis_statistic={self.kurtosis_statistic}, "
            f"  kurtosis_pvalue={self.kurtosis_pvalue}"
            f")"
        )


# helper function for Royston test
# but we will also expose its API to the rest of AnovaPy so it can be used anywhere
def shapiro_francia(
    x: np.typing.ArrayLike,
    *,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> ShapiroFranciaResult:
    """
    Perform the Shapiro-Francia test for normality.

    The Shapiro-Francia test tests the null hypothesis that the data was drawn from a normal distribution.

    Parameters
    ----------
    x : array_like
        Array of sample data (should be a 1D array)
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle NaN input.

    Returns
    -------
    statistic : float
        The W' statistic
    pvalue : float
        The p-value for the hypothesis test

    Notes
    -----
    The test statistic W' is the squared Pearson correlation (r^2) between the ordered sample values and the expected normal order statistics (Blom's approximation). It always lies between [0, 1] with values closer to 1 indicating normality, and values farther from 1 indicate departure from normality.

    The W' statistic itself can be computed for any n >= 2. The constraint
    5 <= n <= 5000 applies to the p-value approximation (Royston, 1993),
    which models ln(1 - W') as approximately normal with mean and standard
    deviation that are polynomial functions of ln(ln n). Outside this range
    the p-value is unreliable.

    This function is not to be confused with the Shapiro-Wilk Test for Normality (see `scipy.stats.shapiro`).
    While both test for normality, the way they function are different. The Shapiro-Wilk statistic W uses optimal
    weights derived from the full nxn covariance matrix of normal order
    statistics. Shapiro-Francia replaces these with the simpler Blom expected normal scores, reducing computation
    to O(n). The two tests are asymptotically equivalent and have comparable
    power for moderate to large samples; Shapiro-Wilk may be slightly more
    powerful for very small samples (n < 30).

    References
    ----------
    .. [1] Shapiro, S. S., & Francia, R. S. (1972). An Approximate Analysis of Variance Test for Normality. Journal of the American Statistical Association, 67(337), 215-216. https://doi.org/10.1080/01621459.1972.10481232
    .. [2] Royston P. A pocket-calculator algorithm for the Shapiro-Francia test for non-normality: an application to medicine. Stat Med. 1993 Jan 30;12(2):181-4. doi: 10.1002/sim.4780120209. PMID: 8446812.
    .. [3] Wikipedia Contributors. "Shapiro-Francia Test." Wikipedia, Wikimedia Foundation, 27 Nov. 2018, https://en.wikipedia.org/wiki/Shapiro%E2%80%93Francia_test. Accessed 11 Nov. 2026.
    """
    x = np.asarray(x)
    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # handle nan
    has_nan = np.isnan(x).any()
    if has_nan:
        if nan_policy == "propagate":
            return ShapiroFranciaResult(statistic=np.nan, pvalue=np.nan)
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        # omit
        x = x[~np.isnan(x)]

    n = len(x)
    x_sorted = np.sort(x)  # sort data to obtain order statistics x_(i)
    # compute expected normal scores
    # formula for this is: m_i = phi^-1((i - 3 / 8) / (n + 1 / 4))
    # where i is the rank order and phi^-1 is the inverse normal cdf
    m = norm.ppf((np.arange(1, n + 1) - 3 / 8) / (n + 1 / 4))
    # now calculate the W'-Statistic
    # this is done by computing the correlation between x_(i) and m_i and squaring it
    W_prime = np.corrcoef(x_sorted, m)[0, 1] ** 2

    # now compute p-value
    # compute Z-score:
    # Z = (ln(1 - W') - u) / standard_deviation
    u = np.log(np.log(n)) - np.log(n)
    v = np.log(np.log(n)) + (2 / np.log(n))
    uu = 1.0521 * u - 1.2725
    sigma = -0.26758 * v + 1.0308
    Z = (np.log(1 - W_prime) - uu) / sigma
    # derive p-value from normal distribution
    p_value = norm.sf(Z)

    return ShapiroFranciaResult(statistic=W_prime, pvalue=p_value)


def henze(
    x: np.typing.ArrayLike,
    *,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> HenzeResult:
    """
    Perform the Henze-Zirkler test for multivariate normality.

    Parameters
    ----------
    x : array_like
        Input data matrix of shape (n_samples, n_features).
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines handling of NaN values.

    Returns
    -------
    HenzeResult
        Dataclass containing the test statistic and asymptotic p-value.

    See Also
    --------
    doornik : Doornik-Hansen Test
    mardia : Mardia's Test
    royston : Royston's Extenion to Shapiro-Wilk
    energy : Energy Test (E-Statistic)

    References
    ----------
    .. [1] Henze, N., and B. Zirkler. "A Class of Invariant Consistent Tests for Multivariate Normality." Communications in Statistics - Theory and Methods, vol. 19, no. 10, Jan. 1990, pp. 3595-617, https://doi.org/10.1080/03610929008830400.
    .. [2] Daniušis, Povilas. Testing Multivariate Normality by Testing Independence. arXiv:2311.11575v2, 23 Dec. 2023, https://arxiv.org/pdf/2311.11575. arXiv. Accessed 18 Oct. 2026.
    .. [3] Korkmaz, Selcuk & Goksuluk, Dincer & Zararsiz, Gokmen. (2014). MVN: An R Package for Assessing Multivariate Normality. The R Journal. 6. 151. 10.32614/RJ-2014-031.
    """
    # Convert input to standard floating-point array
    data = np.asarray(x)

    if data.ndim != 2:
        raise ValueError(
            "Input `x` must be a 2D array of shape (n_samples, n_features)."
        )

    # Handle missing values
    if np.isnan(data).any():
        if nan_policy == "raise":
            raise ValueError("The input contains nan values")
        elif nan_policy == "omit":
            data = data[~np.isnan(data).any(axis=1)]
        elif nan_policy == "propagate":
            return HenzeResult(statistic=np.nan, pvalue=np.nan)
        else:
            raise ValueError(f"Invalid nan_policy: '{nan_policy}'")

    num_obs, num_dim = data.shape

    if num_obs <= num_dim + 1:
        raise ValueError(
            "Sample size must be strictly greater than number of dimensions plus 1."
        )

    # compute means and covariance
    mean_vec = np.mean(data, axis=0)
    cov_mat = np.cov(data, rowvar=False, bias=True)

    # check if covariance matrix is singular
    try:
        inv_cov = np.linalg.inv(cov_mat)
    except np.linalg.LinAlgError:
        # singular covariance matrix fallback or extreme non-normality indicator
        return HenzeResult(statistic=np.inf, pvalue=0.0)

    # center observations
    centered_data = data - mean_vec

    # Mahalanobis distances (individual and pairwise differences)

    # this represents the Squared Mahalanobis distance of the i-th observation from the sample mean
    # or just d_i^2 in short
    dist_sq_single = np.sum((centered_data @ inv_cov) * centered_data, axis=1)

    diff_tensor = data[:, None, :] - data[None, :, :]
    # this represents the Squared Mahalanobis distance between the i-th and j-th observations.
    # or just D_{ij}^2 in short
    dist_sq_pairwise = np.sum((diff_tensor @ inv_cov) * diff_tensor, axis=2)

    # compute smoothing parameter
    smooth_param = (
        (1.0 / np.sqrt(2.0))
        * ((2.0 * num_dim + 1.0) / 4.0) ** (1.0 / (num_dim + 4.0))
        * (num_obs ** (1.0 / (num_dim + 4.0)))
    )
    # the smoothing parameter is squared throughout the formula so we will square it here in advance and reuse it for simplicity
    smooth_sq = smooth_param**2

    # now follow the formula
    # for simplicity we will break it up into parts and put it together

    # part 1: (1 / n^2) * sum_i=1_to_n(sum_j=1_to_n(e ^ (-smooth_param_squared / 2 * D_{ij}^2)))
    part_one = np.sum(np.exp(-0.5 * smooth_sq * dist_sq_pairwise)) / (num_obs**2)
    # part 2: -2/n(1 + smooth_param_squared)^(-p/2) * sum_i=1_to_n(e^-(smooth_param_squared / 2(1 + smooth_param_squared)) * d_{i}^2)
    part_two = (
        2.0
        * ((1.0 + smooth_sq) ** (-num_dim / 2.0))
        * np.mean(np.exp(-(smooth_sq / (2.0 * (1.0 + smooth_sq))) * dist_sq_single))
    )
    # part 3: (1 + 2smooth_param_squared) ^ -p/2
    part_three = (1.0 + 2.0 * smooth_sq) ** (-num_dim / 2.0)

    # finally putting all parts together as according to the formula
    hz_stat = num_obs * (part_one - part_two + part_three)

    # now compute p-value
    # the statistic computed follows the log-normal distribution so we will compare against that
    # the overall formula for calculation of the p-value is p = 1 - norm_cdf(Z)
    # Z can be compute as follows:
    # Z = ln(sqrt(u^4 / (sigma^2 + u^2)))
    # where u and sigma refer to values being computed below
    dim_k = num_dim
    # precompute some parameters in advance as we will reuse the same numbers in the formula again
    # a = 1 + 2B^2
    param_a = 1.0 + 2.0 * smooth_sq
    # w_b = (1 + B^2)(1 + 3B^2)
    param_w = (1.0 + smooth_sq) * (1.0 + 3.0 * smooth_sq)
    smooth_sq2 = smooth_sq**2
    smooth_sq4 = smooth_sq2**2

    # formula for mean here is given as:
    # u = 1 - ((a^-p/2)(1 + pb^2 + (p(p + 2)B^4)) / 2a^2)
    expected_mean = 1.0 - (param_a ** (-dim_k / 2.0)) * (
        1.0
        + (dim_k * smooth_sq) / param_a
        + (dim_k * (dim_k + 2.0) * smooth_sq2) / (2.0 * (param_a**2))
    )

    # now compute sigma^2 (variance) and the formula for this is:
    # sigma^2 = 2(1 + 4B^2)^(-p/2) + ((2a^-p * (1 + 2pB^4)) / a^2) + ((3p(p + 2)B^8) / 4a^4)
    # we will split the computation up into parts since the equation is really huge
    var_comp1 = 2.0 * ((1.0 + 4.0 * smooth_sq) ** (-dim_k / 2.0))
    var_comp2 = (
        2.0
        * (param_a ** (-dim_k))
        * (
            1.0
            + (2.0 * dim_k * smooth_sq2) / (param_a**2)
            + (3.0 * dim_k * (dim_k + 2.0) * smooth_sq4) / (4.0 * (param_a**4))
        )
    )
    var_comp3 = (
        4.0
        * (param_w ** (-dim_k / 2.0))
        * (
            1.0
            + (3.0 * dim_k * smooth_sq2) / (2.0 * param_w)
            + (dim_k * (dim_k + 2.0) * smooth_sq4) / (2.0 * (param_w**2))
        )
    )

    # connect them
    expected_var = var_comp1 + var_comp2 - var_comp3

    if expected_var <= 0:
        p_val = 0.0 if hz_stat > expected_mean else 1.0
    else:
        # log(u)
        ln_scale = np.log(expected_mean**2 / np.sqrt(expected_var + expected_mean**2))
        # log(sigma^2)
        ln_shape = np.sqrt(
            np.log((expected_var + expected_mean**2) / (expected_mean**2))
        )
        p_val = lognorm.sf(hz_stat, ln_shape, scale=np.exp(ln_scale))

    return HenzeResult(statistic=hz_stat, pvalue=p_val)


def doornik(
    x: np.typing.ArrayLike,
    *,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> DoornikResult:
    """
    Runs the Doornik-Hansen test for univariate or multivariate normality.

    The function accepts either a one-dimensional array for a single variable or
    a two-dimensional array for multiple variables.

    Generally speaking, this test compared to Henze-Zirkler, works better when you have small sample sizes.

    Parameters
    ----------
    x : array_like
        Pooled observations. Can be a 1D array or 2D.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines handling of NaN values.

    Returns
    -------
    DoornikResult
        Dataclass containing the test statistic and asymptotic p-value.

    References
    ----------
    .. [1] Doornik, Jurgen A., and Henrik Hansen. "An Omnibus Test for Univariate and Multivariate Normality*." Oxford Bulletin of Economics and Statistics, vol. 70, Dec. 2008, pp. 927-39, https://doi.org/10.1111/j.1468-0084.2008.00537.x. Accessed 19 Oct. 2026.
    .. [2] JDemetra+. "Doornik-Hansen Test." JDemetra+, https://sylwiagrudkowska.github.io/JDemetra-documentation/pages/theory/Tests_DH.html. Accessed 11 Nov. 2026.

    Examples
    --------
    As already stated earlier, this function can work with univariate or multivariate data. The below examples illustrate how.

    >>> from anovapy.normal import doornik
    >>> import numpy as np
    >>> x = np.random.normal(loc=0.0, scale=1.0, size=60)
    >>> doornik_1d_res = doornik(x)
    >>> print(f"Result of univariate data: Statistic is {doornik_1d_res.statistic} and p-value is {doornik_1d_res.pvalue}")
    >>> x = np.random.normal(loc=0.0, scale=1.0, size=(60, 4))
    >>> doornik_2d_res = doornik(x)
    >>> print(f"Result of multivariate data: Statistic is {doornik_2d_res.statistic} and p-value is {doornik_2d_res.pvalue}")
    """
    x = np.asarray(x)

    # Handle 1D arrays (univariate case) by reshaping to column vector
    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.ndim != 2:
        raise ValueError("Input `x` must be a 1D or 2D array.")

    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError(f"Invalid nan_policy: '{nan_policy}'")

    # Handle missing values
    if np.isnan(x).any():
        if nan_policy == "raise":
            raise ValueError("The input contains nan values")
        elif nan_policy == "omit":
            x = x[~np.isnan(x).any(axis=1)]
        else:
            return DoornikResult(statistic=np.nan, pvalue=np.nan)

    n, p = x.shape

    # standardize the data
    if p == 1:
        # with univariate data, no need for decorrelation,
        # standardization, or whitening
        standardized = x
    else:
        X_centered = x - np.mean(x, axis=0)
        std = np.std(X_centered, axis=0, ddof=1)
        x = X_centered / std

        # cov = x.T @ x / (n - 1)
        cov = np.cov(x, rowvar=False, ddof=1)
        evals, evecs = np.linalg.eigh(cov)
        sigma_inv_half = evecs / np.sqrt(evals) @ evecs.T
        standardized = x @ sigma_inv_half

    # compute skew and kurtosis
    s = skew(standardized)
    # note that the formula listed in the paper just says k = m4 / m2 ** 1.5 in short
    # there is no need for that -3 so we set fisher=False to remove that
    k = kurtosis(standardized, fisher=False)

    # compute z1
    beta = (3 * (n**2 + 27 * n - 70) * (n + 1) * (n + 3)) / (
        (n - 2) * (n + 5) * (n + 7) * (n + 9)
    )
    omega = -1 + np.sqrt(2 * (beta - 1))
    delta = 1 / np.sqrt(np.log(np.sqrt(omega)))
    y = s * np.sqrt(((omega - 1) / 2) * ((n + 1) * (n + 3) / (6 * (n - 2))))
    z1 = delta * np.log(y + np.sqrt(y**2 + 1))

    # compute z2
    delta = (n - 3) * (n + 1) * (n**2 + (15 * n) - 4)
    a = ((n - 2) * (n + 5) * (n + 7) * (n**2 + (27 * n) - 70)) / (6 * delta)
    c = ((n - 7) * (n + 5) * (n + 7) * (n**2 + (2 * n) - 5)) / (6 * delta)
    l = ((n + 5) * (n + 7) * (n**3 + (37 * n**2) + (11 * n) - 313)) / (12 * delta)
    alpha = a + c * (s**2)
    chi = (2 * l) * (k - 1 - s**2)
    z2 = np.sqrt(9 * alpha) * ((1 / (9 * alpha)) - 1 + np.cbrt(chi / (2 * alpha)))

    # compute statistic and p-value
    statistic = np.sum(z1**2) + np.sum(z2**2)
    df = 2 * p
    pvalue = chi2.sf(statistic, df)

    return DoornikResult(statistic=statistic, pvalue=pvalue)


def energy(
    x: np.typing.ArrayLike,
    r: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> EnergyResult:
    """
    Does the Energy test (gets the E-statistic).

    Parameters
    ----------
    x : array_like
        Pooled observations. Can be 1D or 2D
    r : int
        Bootstrap resampling. Note that this parameter refers to how many fake datasets you generate. This is used to make the p-value. Generally speaking, the higher the number, the more precise the p-value estimate becomes. Defaults to 1000
    seed : int or np.random.Generator, optional
        Seed or random number generator used for generating random normally distributed datasets to help compute the p-value.
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle NaNs in input arrays.

    Returns
    -------
    statistic : float
        The E-Statistic
    pvalue : float
        The computed p-value

    Notes
    -----
    This test should generally be used instead of moment-based methods like Mardia's Test when the data may contain heavy-tailed distributions or when you need a non-parametric approach that does not rely on estimating skewness and kurtosis. It serves as a robust alternative to classical tests, offering computational efficiency and strong power across a wide range of deviations from Gaussianity, making it suitable for complex datasets in fields like genomics or finance.

    A word on the `r` parameter. It is very important to understand how this test works and the p-value computation. The core idea of the test is straightforward: *You want to know "how extreme is my observed statistic, if the data really were normal?" You can't derive that distribution analytically (it's a nasty weighted sum of chi-squares), so you **simulate** it: pretend the null is true, generate fake normal data, recompute the statistic, and see where your real statistic falls.*

    That's it. So first, the E-statistic gets computed (we will refer to this as E_obs for simplicity), then multivariate normal data (this should have the exact same shape as x) is generated `r` times and compute the E-statistic on those. Your p-value = the fraction of fake statistics that are at least as large as E_obs.

    References
    ----------
    .. [1] Székely, Gábor J., and Maria L. Rizzo. "A New Test for Multivariate Normality." Journal of Multivariate Analysis, vol. 93, no. 1, Elsevier BV, Mar. 2005, pp. 58-80, https://doi.org/10.1016/j.jmva.2003.12.002.
    .. [2] Maria L. Rizzo and Gabor J. Szekely. "E-Statistic (Energy) Test of Multivariate Normality." Energy, https://mariarizzo.github.io/energy/reference/mvnorm-test.html. Accessed 11 Nov. 2026.

    Examples
    --------
    >>> from anovapy.normal import energy
    >>> import numpy as np
    >>> sample_multivariate = np.random.normal(size=(60, 4))
    >>> sample_univariate = np.random.normal(size=100)
    >>> multivariate_res = energy(sample_multivariate)
    >>> print(f"Statistic: {multivariate_res.statistic}, p-value: {multivariate_res.pvalue}")
    >>> univariate_res = energy(sample_univariate)
    >>> print(f"Statistic: {univariate_res.statistic}, p-value: {univariate_res.pvalue}")
    >>> # changing r and seed parameter
    >>> res = energy(sample_multivariate, r=500, seed=42)
    >>> print(f"Statistic: {res.statistic}, p-value: {res.pvalue}")
    """
    x = np.asarray(x)

    # Handle 1D arrays (univariate case) by reshaping to column vector
    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif x.ndim != 2:
        raise ValueError("Input `x` must be a 1D or 2D array.")

    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError(f"Invalid nan_policy: '{nan_policy}'")

    # Handle missing values
    if np.isnan(x).any():
        if nan_policy == "raise":
            raise ValueError("The input contains nan values")
        elif nan_policy == "omit":
            x = x[~np.isnan(x).any(axis=1)]
        else:
            return EnergyResult(statistic=np.nan, pvalue=np.nan)

    def stat_compute(X: np.ndarray):
        n, d = X.shape

        # --- Standardize to zero mean, identity covariance ---
        if d == 1:
            # standardize univariate data
            x_bar = X.mean(axis=0)
            std = X.std(axis=0, ddof=1)
            Y = (X - x_bar) / std
        else:
            x_bar = X.mean(axis=0)
            S = np.cov(X, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(S)
            S_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            Y = (X - x_bar) @ S_inv_sqrt  # shape (n, d); row i = y_i

        # --- Term 1: (2/n) * sum_i E||y_i - Z|| ---
        ysq = np.sum(Y**2, axis=1)  # ||y_i||^2 for each i, shape (n,)
        const = gamma((d + 1) / 2) / gamma(d / 2)
        mean1 = np.mean(np.sqrt(2) * const * hyp1f1(-0.5, d / 2, -ysq / 2))

        # --- Term 2: E||Z - Z'|| (closed-form constant) ---
        mean2 = 2 * const

        # --- Term 3: (1/n^2) * sum_i sum_j ||y_i - y_j|| ---
        D = squareform(pdist(Y))  # full (n, n) distance matrix
        mean3 = D.sum() / n**2

        return n * (2 * mean1 - mean2 - mean3)

    stat = stat_compute(x)

    n, d = x.shape

    rng = np.random.default_rng(seed)

    check = np.array([stat_compute(rng.standard_normal(size=(n, d))) for _ in range(r)])

    p_value = 1 - np.mean(check < stat)

    return EnergyResult(statistic=stat, pvalue=p_value)


def royston(
    x: np.typing.ArrayLike,
    *,
    method: Literal["auto", "shapiro", "francia"] = "auto",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> RoystonResult:
    """
    Does the Royston Multivariate Extension for Shapiro-Wilk Normliaty Test.

    Parameters
    ----------
    x : array_like
        Array of sample data. Must contain at least three observations. It must also be of shape (n_samples, n_features)
    method : {'auto', 'shapiro', 'francia'}, default='auto'
        The per-variable normality statistic used to compute H.

        'shapiro'
            Always uses the Shapiro-Wilk W statistic, as described in the
            original Royston (1992) paper [1].

        'francia'
            Always uses the Shapiro-Francia W' statistic.

        'auto' (default)
            Uses the Shapiro-Francia W' statistic for variables with
            kurtosis > 3 (leptokurtic), and Shapiro-Wilk W otherwise.
            This matches the behavior of the R MVN package [2], which
            applies this heuristic because Shapiro-Francia has greater
            power against heavy-tailed alternatives.

    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle input NaNs.

    Returns
    -------
    statistic : float
        The h-statistic
    pvalue : float
        The p-value for this test

    References
    ----------
    .. [1] Royston, P. Approximating the Shapiro-Wilk W-test for non-normality. Stat Comput 2, 117-119 (1992). https://doi.org/10.1007/BF01891203
    .. [2] Farrell, Patrick J., et al. "On Tests for Multivariate Normality and Associated Simulation Studies." Journal of Statistical Computation and Simulation, vol. 77, no. 12, Dec. 2007, pp. 1065-80, https://doi.org/10.1080/10629360600878449.
    .. [3] Korkmaz, Selcuk & Goksuluk, Dincer & Zararsiz, Gokmen. (2014). MVN: An R Package for Assessing Multivariate Normality. The R Journal. 6. 151. 10.32614/RJ-2014-031.

    Examples
    --------
    >>> from anovapy.normal import royston
    >>> from scipy.stats import multivariate_normal
    >>> import numpy as np
    >>> # generate 10 random samples
    >>> cov = [[2.0, 0.3], [0.3, 0.5]]
    >>> samples = multivariate_normal.rvs(mean=[1, 2], cov=cov, size=10)
    >>> res = royston(samples)
    >>> print(f"H-Statistic: {res.statistic}, p-value: {res.pvalue}")
    """
    # convert to array as needed
    x = np.asarray(x)

    # validate dimension
    if x.ndim != 2:
        # oops
        raise ValueError("An array of shape (n_samples, n_features) must be passed!")

    # handle nan
    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'.")
    has_nan = any(np.isnan(sample).any() for sample in x)
    if has_nan and nan_policy == "raise":
        raise ValueError("The input contains nan values")
    if has_nan and nan_policy == "propagate":
        return RoystonResult(statistic=np.nan, pvalue=np.nan)
    if has_nan:
        keep = ~np.isnan(x[0]).any(axis=1)
        x[0] = x[0][keep]

    # handle limits
    n, p = x.shape

    if n < 3 or n > 2000:
        raise ValueError(
            "Royston's test only works for sample sizes between 3 and 2000."
        )

    # calculate Shapiro-Wilk w stat for all and transform them
    z_transformed = np.zeros(p)

    if n >= 12:
        v = np.log(n)
        m = -1.5861 - 0.31082 * v - 0.083751 * v**2 + 0.0038915 * v**3
        s = np.exp(-0.4803 - 0.082676 * v + 0.0030302 * v**2)
    else:  # 4 <= n <= 11
        g = -2.273 + 0.459 * n
        m = 0.5440 - 0.39978 * n + 0.025054 * n**2 - 0.0006714 * n**3
        s = np.exp(1.3822 - 0.77857 * n + 0.062767 * n**2 - 0.0020322 * n**3)

    for j in range(p):
        # what is the method for computing the stat?
        w_stat: float = 0.0  # to be set
        col = x[:, j]
        if method == "auto":
            w_stat, _ = shapiro(col)
            if kurtosis(col, fisher=False) > 3:
                w_stat, _ = shapiro_francia(col)
        elif method == "shapiro":
            # only shapiro-wilk is used
            w_stat, _ = shapiro(col)
        else:
            # only shapiro-francia is used
            # this is more rare by the way
            w_stat, _ = shapiro_francia(col)
        # apply transformation
        if n >= 12:
            z_transformed[j] = (np.log(1.0 - w_stat) - m) / s
        else:
            z_transformed[j] = (-np.log(g - np.log(1.0 - w_stat)) - m) / s

    # now transform z transformations to chi-squared
    x2 = (norm.ppf(norm.cdf(-z_transformed) / 2.0)) ** 2

    # compute effective degrees of freedom
    u = 0.715
    v = 0.21364 + 0.015124 * np.log(n) ** 2 - 0.0018034 * np.log(n) ** 3
    l = 5

    C = np.corrcoef(x, rowvar=False)  # pxp correlation matrix
    NC = (C**l) * (1.0 - (u * (1.0 - C) ** u) / v)  # transformed corr matrix

    T = NC.sum() - p
    mC = T / (p**2 - p)  # average correlation
    e = p / (1.0 + (p - 1) * mC)  # effective degrees of freedom

    # now compute h-stat and p-value
    H = e * x2.sum() / p
    p_value = chi2.sf(H, e)

    return RoystonResult(H, p_value)


def mardia(
    x: np.typing.ArrayLike,
    *,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> MardiaResult:
    """
    Runs Mardia's Test for Normality

    Parameters
    ----------
    x : array_like
        Pooled observations
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        How to handle NaN values in observations. ``omit`` removes any
        observation with a NaN response; ``raise`` raises ``ValueError``.
        Missing group labels are always an error.

    Returns
    -------
    MardiaResult
        The statistics and p-values of skewness and kurtosis. Generally speaking, if the p-value of kurtosis and skewness are > 0.05 (or whatever your threshold is), you can assume normality.

    Raises
    ------
    ValueError
        If the input contains any NaN values and the `nan_policy` is set to `raise`

    References
    ----------
    .. [1] Mardia, K. V. (1970). Measures of Multivariate Skewness and Kurtosis with Applications. Biometrika, 57(3), 519-530. https://doi.org/10.2307/2334770
    .. [2] Zaiontz, Charles. "Multivariate Normality Testing (Mardia)." Real-Statistics.Com, 2024, https://real-statistics.com/multivariate-statistics/multivariate-normal-distribution/multivariate-normality-testing/. Accessed 11 Nov. 2026.
    """
    x = np.asarray(x)
    # deal with nan stuff
    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'.")
    has_nan = any(np.isnan(sample).any() for sample in x)
    if has_nan and nan_policy == "raise":
        raise ValueError("The input contains NaN values.")
    if has_nan and nan_policy == "propagate":
        return MardiaResult(
            skew=np.nan,
            skew_statistic=np.nan,
            skew_pvalue=np.nan,
            kurtosis=np.nan,
            kurtosis_statistic=np.nan,
            kurtosis_pvalue=np.nan,
        )
    if has_nan:
        keep = ~np.isnan(x[0]).any(axis=1)
        x[0] = x[0][keep]

    # technically mardia's test doesn't actually work with 1D arrays
    # so just transform it into 2d and give a warning
    # if X.ndim == 1:
    #     _mardia_logger.warning('WARNING: a 1D array was passed into the mardia() function. Mardia\'s test should not be used on 1D arrays -- use Shapiro-Wilk instead (see scipy.stats.shapiro).')
    #     # transform
    #     X = np.reshape(X, (-1, 2), order='F')
    if x.ndim != 2:
        raise ValueError("An array of shape (n_samples, n_features) must be passed!")
    n, p = x.shape

    # compute nean vector and covariance matrix
    mean = np.mean(x, axis=0)
    cov = np.cov(x, rowvar=False)
    inv_cov = np.linalg.inv(cov)

    # center the data
    X_centered = x - mean

    # now compute the Mahalanobis distances squared
    # formula for this is: d_ij = (x_i - mean)^T * S^-1 * (x_j - mean)
    mah_dist = np.diag(X_centered @ inv_cov @ X_centered.T)

    # compute full distance matrix for skewness
    D = X_centered @ inv_cov @ X_centered.T

    # finally compute Mardia's multivariate skewness (b1,p)
    skewness = np.sum(D**3) / (n**2)

    # compute skew statistic
    # formula for this is: (n / 6) * (skewness)
    # this makes a Chi-square test statistic which can be used against the chi distribution to compute the p-value
    skew_stat = (n * skewness) / 6.0
    # compute degrees of freedom
    df = (p * (p + 1) * (p + 2)) / 6
    skew_pvalue = chi2.sf(skew_stat, df)

    # compute Mardia's multivariate kurtosis (b2,p)
    kurtosis = np.sum(mah_dist**2) / n

    # compute kurtosis statistic
    # formula for this is: (kurtosis - p(p + 2)) / sqrt(8p(p+2) / n)
    kurt_stat = (kurtosis - p * (p + 2)) / np.sqrt((8 * p * (p + 2)) / n)
    # now compute p-value
    # this is compared against a standard normal distribution and is computed as a two-tailed test
    kurt_pvalue = 2 * norm.sf(np.abs(kurt_stat))

    # if p-value of kurtosis and skew are both > 0.05 (or whatever the threshold is), you can assume normality
    return MardiaResult(
        skew=skewness,
        skew_statistic=skew_stat,
        skew_pvalue=skew_pvalue,
        kurtosis=kurtosis,
        kurtosis_statistic=kurt_stat,
        kurtosis_pvalue=kurt_pvalue,
    )
