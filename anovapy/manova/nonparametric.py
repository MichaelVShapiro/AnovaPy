# this contains a non-parametric inference for Multivariate Data (i.e. Nonparametric MANOVA)

from collections import namedtuple
from typing import Literal

import numpy as np
from scipy import linalg
from scipy.stats import f, rankdata

from .core import _f_approximation  # we will reuse this function

__all__ = ["nonparametric_manova"]

NonParametricManovaResult = namedtuple(
    "NonParametricManovaResult", ["statistic", "pvalue"]
)


def nonparametric_manova(
    dependent: np.typing.ArrayLike,
    independent: np.typing.ArrayLike,
    *,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> NonParametricManovaResult:
    """
    Runs a Non-Parametric MANOVA.

    This is used when assumptions are not met for a One Way MANOVA.

    Parameters
    ---------
    dependent : array_like
        Pooled observations with shape (n_observations, n_variables). Each
        column is ranked independently across all observations.
    independent : array_like
        A one-dimensional array containing the group label for each
        observation.
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle NaN values in the dependent data. With ``omit``,
        observations with NaN group labels are omitted as well.

    Returns
    -------
    statistic : float
        The Hotelling-Lawley F-approximation statistic for the ranked data.
    pvalue : float
        Upper-tail probability from the approximate F distribution.

    See Also
    --------
    multivariate_kruskal : Multivariate Kruskal-Wallis Test
    permanova : Permuational Analysis of Variance (PERMANOVA)

    Notes
    -----
    After ranking each variable, the test partitions its rank sums of squares
    and cross-products into a between-group hypothesis matrix, H, and a
    within-group error matrix, E:

    H = sum_g n_g (mean_rank_g - mean_rank)(mean_rank_g - mean_rank)^T

    E = sum_g sum_i (rank_ig - mean_rank_g)(rank_ig - mean_rank_g)^T

    From there, an ANOVA-type statistic is computed. This function implements the Lawley-Hotelling statistic, which is defined by the formula:

    $U = tr(H * E^-1)$

    The *U* statistic is then converted into the F-statistic and the p-value is estimated upon the F-distribution from there.

    Note that this function currently only implements the Lawley-Hotelling statistic and returns the converted F-statistic with its p-value.

    This rank-based test is an alternative to one-way MANOVA when its
    distributional assumptions are not suitable. The F distribution is an
    approximation, not an exact finite-sample distribution.

    References
    ----------
    [1] Burchett, W. W., Ellis, A. R., Harrar, S. W., & Bathke, A. C. (2017). Nonparametric Inference for Multivariate Data: The R Package npmv. Journal of Statistical Software, 76(4), 1-18. https://doi.org/10.18637/jss.v076.i04
    [2] Arne C. Bathke , Solomon W. Harrar, and Laurence V. Madden. "How to compare small multivariate samples using nonparametric tests," Computational Statistics and Data Analysis 52 (2008) 4951-4965
    """
    dependent = np.asarray(dependent)
    independent = np.asarray(independent)

    # validate everything as always
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # accept array-like group labels with an extra singleton dimension, but reject inputs that still contain more than one dimension after squeezing.
    if independent.ndim > 1:
        independent = independent.squeeze()
        if independent.ndim > 1:
            raise ValueError("Independent variable must be 1-dimensional.")

    # the matrix operations below treat rows as observations and columns as separate dependent variables.
    if dependent.ndim != 2:
        raise ValueError(
            "Dependent variable must be a 2D array-like of shape (n_samples, n_features)."
        )

    if len(dependent) != len(independent):
        raise ValueError(
            "Dependent and independent arrays must have the same length along axis 0."
        )

    # handle nan
    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        elif nan_policy == "propagate":
            return NonParametricManovaResult(statistic=np.nan, pvalue=np.nan)
        elif nan_policy == "omit":
            valid_mask = ~np.isnan(dependent).any(axis=1) & ~np.isnan(independent)
            dependent = dependent[valid_mask]
            independent = independent[valid_mask]

    # we can already define N in the formula since it just refers to the number of observations
    # p just refers to the number of variables which we will also use later
    N, p = dependent.shape

    # get number of (unique) labels
    groups = np.unique(independent)
    g = len(groups)

    # validate of course
    if g < 2:
        raise ValueError("Independent variable must contain at least 2 unique groups.")
    if N <= g:
        raise ValueError(
            "Number of observations must be greater than the number of groups."
        )

    # now rank the data
    ranks = np.empty_like(dependent)
    for j in range(p):
        ranks[:, j] = rankdata(dependent[:, j], method="average")

    # now calculate overall mean rank vector and group mean rank vectors (very important for measuring between-group differences)
    overall_mean_rank = ranks.mean(axis=0)

    # we need to compute H and E now

    # H is the between-group hypothesis SSCP matrix. Each group's rank-mean
    # deviation contributes an outer product, weighted by that group's size;
    # this captures both the size and direction of the group effect.
    H = np.zeros((p, p), dtype=np.float64)  # to be filled

    # E is the within-group error SSCP matrix. Centering each observation on
    # its own group's rank mean isolates variation not explained by group.
    E = np.zeros((p, p), dtype=np.float64)  # to be filled
    for group in groups:
        group_ranks = ranks[independent == group]
        group_mean_rank = group_ranks.mean(axis=0)
        mean_difference = group_mean_rank - overall_mean_rank
        H += len(group_ranks) * np.outer(mean_difference, mean_difference)

        centered_ranks = group_ranks - group_mean_rank
        E += centered_ranks.T @ centered_ranks

    # rank variation decomposes as total = between + within. The generalized
    # eigenvalues of (H, total) are the MANOVA canonical roots; transforming
    # each root r to r/(1-r) gives the corresponding root of H E^-1. Their
    # sum is therefore the Lawley-Hotelling trace tr(H E^-1).
    total = H + E
    try:
        roots = linalg.eigvalsh(H, total)
    except linalg.LinAlgError as exc:
        # a singular total rank SSCP matrix prevents the generalized
        # eigenvalue problem from defining the statistic
        raise ValueError("The rank sums-of-squares matrix is singular.") from exc

    # eigenvalues are theoretically real and between zero and one; clamp only
    # floating-point excursions at the boundaries before evaluating r/(1-r).
    roots = np.clip(np.real(roots), 0.0, np.nextafter(1.0, 0.0))
    # this would be our U statistic by the way
    lawley_hotelling = np.sum(roots / (1.0 - roots))

    # lets reuse the F-approximation function already implemented earlier to transform the U statistic into an F-statistic and then compute the p-value since it already has the formula implemented for the transformation anyways
    statistic, df_num, df_den = _f_approximation(
        lawley_hotelling,
        roots,
        p,
        g - 1,
        N - g,
        "hotelling_lawley",
    )

    # now compute the p-value based on the adjusted F-statistic
    p_value = f.sf(statistic, df_num, df_den)

    return NonParametricManovaResult(statistic=statistic, pvalue=p_value)
