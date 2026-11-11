# Multivariate Kruskal-Wallis test

from collections import namedtuple
from typing import Literal

import numpy as np
from scipy.stats import chi2, kruskal, rankdata

__all__ = ["multivariate_kruskal", "univariate_kruskal"]

MultivariateKruskalResult = namedtuple(
    "MultivariateKruskalResult", ["statistic", "pvalue"]
)
UnivariateKruskalResult = namedtuple("UnivariateKruskalResult", ["statistic", "pvalue"])


def multivariate_kruskal(
    dependent: np.typing.ArrayLike,
    independent: np.typing.ArrayLike,
    *,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> MultivariateKruskalResult:
    """
    Runs the Multivariate Kruskal-Wallis test.

    This is used when assumptions are not met for a One Way MANOVA.

    Parameters
    ---------
    dependent : array_like
        Pooled observations. Should be of shape (n_observations, n_variables)
    independent : array_like
        A 1D array of independent variables
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle NaN inputs.

    Returns
    -------
    statistic : float
        The computed L-statistic
    pvalue : float
        Computed p-value from the Chi-squared distribution.

    Notes
    -----
    The general formula for this test is:

    $L = (N - 1) * tr(H * E^-1)$

    where *N* is the total number of participants/observations combined across all groups, *H* (hypothesis matrix) is the between-groups sum of squares and cross-products of the ranks, and *E* (error matrix) is the total sum of squares and cross-products of the ranks (or sometimes the within-group error, depending on the exact formulation).

    This formula is what also computes the Lawley-Hotelling Trace. Comparing it to a MANOVA, a Multivariate Kruskal-Wallis takes this formula and applies it to ranks essentially.

    Note that this test is specifically supposed to be an alternative for a One Way MANOVA.

    References
    ----------
    .. [1] Katz BM, McSweeney M. A Multivariate Kruskal-Wallis Test With Post Hoc Procedures. Multivariate Behav Res. 1980 Jul 1;15(3):281-97. doi: 10.1207/s15327906mbr1503_4. PMID: 26794183.
    .. [2] Puri, Madan Lal, and Pranab Kumar Sen. "A Class of Rank Order Tests for a General Linear Hypothesis." The Annals of Mathematical Statistics, vol. 40, no. 4, Institute of Mathematical Statistics, Aug. 1969, pp. 1325-43, https://doi.org/10.1214/aoms/1177697505. Accessed 8 Nov. 2026.
    """
    dependent = np.asarray(dependent)
    independent = np.asarray(independent)

    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # also validate data
    # handle shape conversions / flattening for independent variable
    if independent.ndim > 1:
        independent = independent.squeeze()
        if independent.ndim > 1:
            raise ValueError("Independent variable must be 1-dimensional.")

    if dependent.ndim != 2:
        raise ValueError(
            "Dependent variable must be a 2D array-like of shape (n_samples, n_features)."
        )

    if len(dependent) != len(independent):
        raise ValueError(
            "Dependent and independent arrays must have the same length along axis 0."
        )

    # now handle nan input
    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        elif nan_policy == "propagate":
            return MultivariateKruskalResult(statistic=np.nan, pvalue=np.nan)
        elif nan_policy == "omit":
            valid_mask = ~np.isnan(dependent).any(axis=1) & ~np.isnan(independent)
            dependent = dependent[valid_mask]
            independent = independent[valid_mask]

    # we can already define N in the formula since it just refers to the number of observations
    N, p = dependent.shape

    # get number of independent groups
    groups = np.unique(independent)
    g = len(groups)

    if g < 2:
        raise ValueError("Independent variable must contain at least 2 unique groups.")
    if N <= g:
        raise ValueError(
            "Number of observations must be greater than the number of groups."
        )

    # first rank the data
    ranks = np.empty_like(dependent)
    for j in range(p):
        ranks[:, j] = rankdata(dependent[:, j], method="average")

    # now calculate overall mean rank vector and group mean rank vectors
    overall_mean_rank = ranks.mean(axis=0)

    # we need to compute H and E here

    # lets compute the Error Matrix (E)
    # this is defined to be E = sum((R_i - R) * (R_i - R)^T)
    # take the sum of squares and cross products of ranks simply
    diff_total = ranks - overall_mean_rank
    E = diff_total.T @ diff_total

    # H (hypothesis matrix) is defined to be: H = sum(n_j * (R_j - R) * (R_j - R)^T)
    # we just need to take between groups sum of squares and cross products here
    H = np.zeros((p, p))  # to be filled
    for i in groups:
        mask = independent == i
        n_j = np.sum(mask)
        if n_j == 0:
            continue
        group_mean_rank = ranks[mask].mean(axis=0)
        diff = group_mean_rank - overall_mean_rank
        H += n_j * np.outer(diff, diff)

    # now put it all together to compute the test statistic
    L = (N - 1) * np.linalg.trace(H @ np.linalg.inv(E))

    # compute degrees of freedom
    df = p * (g - 1)  # this is the numerator degrees of freedom

    # now calculate the p-value
    p_value = chi2.sf(L, df)

    return MultivariateKruskalResult(statistic=L, pvalue=p_value)


def univariate_kruskal(
    dependent: np.typing.ArrayLike,
    independent: np.typing.ArrayLike,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> UnivariateKruskalResult:
    """
    Perform a Univariate Kruskal-Wallis test on each dependent variable.

    This is typically used as a post-hoc test after a multivariate test like MANOVA or Multivariate Kruskal-Wallis test.

    Parameters
    ----------
    dependent : array-like, shape (n_samples, n_features)
        The data to be tested.
    independent : array-like, shape (n_samples,)
        The group labels for each sample.
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle NaN inputs.

    Returns
    -------
    statistic : np.ndarray
        The Kruskal-Wallis H statistics for each comparison.
    pvalue : np.ndarray
        The p-values for each comparison.

    Notes
    -----
    This is **not** a true multivariate test. Instead, it is commonly used as an
    exploratory post-hoc or follow-up after a significant global multivariate
    non-parametric test (such as a Multivariate Kruskal-Wallis test). Because multiple
    independent tests are performed, the p-value computed should be adjusted in some way such as the Bonferroni method to control the familywise error rate.

    References
    ----------
    .. [1] Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. Journal of the American Statistical Association, 47(258), 583-621.
    .. [2] Hollander, M., Wolfe, D. A., & Chicken, E. (2013). Nonparametric Statistical Methods. John Wiley & Sons.
    """
    dependent = np.asarray(dependent)
    independent = np.asarray(independent)

    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    if independent.ndim > 1:
        independent = independent.squeeze()
        if independent.ndim > 1:
            raise ValueError("Independent variable must be 1-dimensional.")

    if dependent.ndim != 2:
        raise ValueError(
            "Dependent variable must be a 2D array-like of shape (n_samples, n_features)."
        )

    if len(dependent) != len(independent):
        raise ValueError(
            "Dependent and independent arrays must have the same length along axis 0."
        )

    # handle nan input
    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        elif nan_policy == "propagate":
            return UnivariateKruskalResult(statistic=np.nan, pvalue=np.nan)
        elif nan_policy == "omit":
            valid_mask = ~np.isnan(dependent).any(axis=1) & ~np.isnan(independent)
            dependent = dependent[valid_mask]
            independent = independent[valid_mask]

    unique_labels = np.unique(independent)
    grouped_data = [dependent[independent == label] for label in unique_labels]

    # just perform the Kruskal-Wallis H Test for each feature
    statistics = []
    pvalues = []

    for feature in range(dependent.shape[1]):
        feature_data = [group[:, feature] for group in grouped_data]
        stat, pval = kruskal(*feature_data, nan_policy=nan_policy)
        statistics.append(stat)
        pvalues.append(pval)

    return UnivariateKruskalResult(
        statistic=np.array(statistics), pvalue=np.array(pvalues)
    )
