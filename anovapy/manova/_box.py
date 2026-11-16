# Box's M-Test for checking homogeneity variance a critical assumption of MANOVA

from collections import namedtuple
from typing import Literal

import numpy as np
from scipy.stats import chi2

__all__ = ["box"]

BoxResult = namedtuple("BoxResult", ["statistic", "pvalue"])


def box(
    x: np.typing.ArrayLike,
    groups: np.typing.ArrayLike,
    *,
    correction: bool = True,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> BoxResult:
    """
    Box's M-Test for checking homogeneity of covariance matrices across two or more groups.

    Parameters
    ----------
    x : array_like
        Pooled observations. Must be of shape (n_observations, n_variables)
    groups : array_like
        A 1D array of representing group labels for each observation.
    correction : bool
        Whether to apply a correction factor to the M-Statistic. Default to True
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle NaN inputs

    Returns
    -------
    statistic : float
        The M-statistic
    pvalue : float
        Computed p-value from the Chi-Square Distribution

    Notes
    -----
    The general formula to compute the M-statistic is:

    $M = (N - k)ln|S_p| - sum((n_i - 1) * ln|S_i|)$

    where *N* is the total sample size, *k* is the number of groups, *n_i* is the sample size of group i, *S_i* is the sample covariance matrix for group i, and *S_p* is the pooled covariance matrix.

    Sometimes a correction factor may be applied to this statistic to compute the p-value, for which the formula for that is:

    $M_adj = (1 - C)M$

    where *C* can be computed by the formula:

    $C = ((2p^2 + 3p - 1) / (6(p + 1)(k - 1))) / (sum((1 / (n_i - 1)) - (1 / (N - k))))$

    This is sometimes done to provide a more accurate p-value for small sample sizes.

    References
    ----------
    .. [1] BOX, G. E. P. "A GENERAL DISTRIBUTION THEORY for a CLASS of LIKELIHOOD CRITERIA." Biometrika, vol. 36, no. 3-4, 1949, pp. 317-46, https://doi.org/10.1093/biomet/36.3-4.317.
    .. [2] Zaiontz, Charles. "Box's M Test Basic Concepts." Real-Statistics.Com, 2014, https://real-statistics.com/multivariate-statistics/boxs-test/boxs-test-basic-concepts/. Accessed 15 Nov. 2026.
    """
    x = np.asarray(x, dtype=float)
    groups = np.asarray(groups)

    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # handle nan
    if np.isnan(x).any():
        if nan_policy == "propagate":
            return BoxResult(statistic=np.nan, pvalue=np.nan)
        elif nan_policy == "raise":
            raise ValueError("The input contains NaN values.")
        x = x[~np.isnan(x).any(axis=1)]

    N, p = x.shape

    # get sizes for each group
    unique_groups = np.unique(groups)
    k = len(unique_groups)

    # now compute sample covariance matrix for each group
    S_i = []
    n_i = []  # sizes for each group
    for i in unique_groups:
        # get all items part of that group
        mask = groups == i
        items = x[mask]
        n_i.append(len(items))
        S_i.append(np.cov(items, rowvar=False))
    S_i = np.array(S_i)
    n_i = np.array(n_i)

    # now compute the pooled covariance matrix
    S_p = np.sum([(ni - 1) * Si for ni, Si in zip(n_i, S_i)], axis=0) / (N - k)

    # compute the log determinants (this refers to the ln|x| part of the equation)
    _, S_p_log = np.linalg.slogdet(S_p)
    S_i_log = [np.linalg.slogdet(i)[1] for i in S_i]

    # finally put it all together
    M = (N - k) * S_p_log - np.sum((n_i - 1) * S_i_log)

    # compute degrees of freedom
    df = p * (p + 1) * (k - 1) / 2

    # apply correction optionally
    if correction:
        # the equation here is huge let's split it up
        pt1 = (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (k - 1))
        pt2 = np.sum(1 / (n_i - 1)) - 1 / (N - k)
        C = pt1 * pt2
        M = (1 - C) * M

    # now compute p-value
    p_value = chi2.sf(M, df)

    return BoxResult(statistic=M, pvalue=p_value)
