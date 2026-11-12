# implementation of PERMANOVA (permutational multivariate analysis of variance)

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.distance import pdist, squareform

__all__ = ["PERMANOVAResult", "permanova"]


@dataclass(frozen=True)
class PERMANOVAResult:
    """
    Results of the Permutational Analysis of Variance (PERMANOVA).

    Attributes
    ----------
    statistic : float
        Computed F-Statistic (observed)
    pvalue : float
        Computed p-value
    eta_squared : float
        Computed (observed) effect size
    dof_between : int
        Degrees of Freedom (between groups)
    dof_within : int
        Degrees of Freedom (residual)
    """

    statistic: float
    pvalue: float
    eta_squared: float
    dof_between: int
    dof_within: int

    def __repr__(self) -> str:
        return (
            f"PERMANOVAResult("
            f"  statistic={self.statistic:.6f}, "
            f"  pvalue={self.pvalue:.6f}, "
            f"  eta_squared={self.eta_squared:.6f}, "
            f"  dof_between={self.dof_between}, "
            f"  dof_within={self.dof_within}"
            f")"
        )


def _validate_nan_policy(
    X: np.typing.ArrayLike,
    group_labels: np.typing.ArrayLike,
    nan_policy: Literal["propagate", "omit", "raise"],
) -> tuple[np.typing.ArrayLike, np.typing.ArrayLike]:
    """Handle missing values in feature matrix X and sync with group_labels."""
    has_nan = np.isnan(X).any()
    if has_nan:
        if nan_policy == "raise":
            raise ValueError("Input feature matrix contains NaN values.")
        elif nan_policy == "omit":
            valid_mask = ~np.isnan(X).any(axis=1)
            return X[valid_mask], group_labels[valid_mask]
        elif nan_policy == "propagate":
            return X, group_labels
    return X, group_labels


def compute_pseudo_f(
    dist_sq: np.typing.ArrayLike,
    group_labels: np.typing.ArrayLike,
    unique_groups: np.typing.ArrayLike,
    n_samples: int,
) -> tuple[float, float, float, int, int]:
    """
    Calculate Pseudo-F statistic and Eta-Squared from a squared distance matrix as described in the paper (see [1] in `anovapy.manova.permanova`).

    The formula here is:

    F = (SS_between / (g - 1)) / (SS_within / (n - g))

    where *SS_between* refers to the Sum of Squares between groups, *SS_within** refers to Sum of Squares within groups (residual), *g* is the number of groups, and *n* is the total number of samples.
    """
    n_groups = len(unique_groups)

    # SS_total = sum(d_ij^2) / N
    ss_t = np.sum(dist_sq) / (2.0 * n_samples)

    # SS_within = sum( sum(d_group^2) / n_group )
    ss_w = 0.0
    for g in unique_groups:
        mask = group_labels == g
        n_g = np.sum(mask)
        if n_g > 1:
            sub_dist_sq = dist_sq[np.ix_(mask, mask)]
            ss_w += np.sum(sub_dist_sq) / (2.0 * n_g)

    ss_a = ss_t - ss_w
    dof_between = n_groups - 1
    dof_within = n_samples - n_groups

    if ss_w == 0 or dof_within <= 0:
        f_stat = np.nan
    else:
        f_stat = (ss_a / dof_between) / (ss_w / dof_within)

    # effect size = ss_effect / ss_total
    eta_squared = ss_a / ss_t if ss_t > 0 else 0.0
    return f_stat, eta_squared, ss_t, dof_between, dof_within


def permanova(
    dependent: np.typing.ArrayLike,
    independent: np.typing.ArrayLike,
    *,
    permutations: int = 999,
    distance_metric: str | callable = "euclidean",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
    random_state: int | np.random.Generator | None = None,
) -> PERMANOVAResult:
    """
    Perform a One-Way Permutational Multivariate Analysis of Variance (PERMANOVA).

    This is a non-parametric test used when assumptions of MANOVA are not met (normality and homogeneity of variance). It tests whether the centroids of groups are significantly different in a multivariate space.

    Parameters
    ----------
    dependent : array-like
        A pooled observation matrix, or the first of two separate group
        matrices. Rows are observations and columns are features.
    independent : array-like, optional
        A 1D array of group labels for the pooled observations, or the second
        group matrix. When supplied as labels, there must be one label per
        observation. Use `grouping` as a keyword alias if desired.
    permutations : int, default=999
        The number of permutations to perform for the permutation test.
    distance_metric : str or callable, default='euclidean'
        The distance metric to use for computing the distance matrix. Can be any metric supported by `scipy.spatial.distance.pdist` or a custom callable.
    nan_policy : {'propagate', 'omit', 'raise'}, default='propagate'
        Defines how to handle NaN values in the input data:
        - 'propagate': NaNs are kept, and the result will contain NaNs.
        - 'omit': Rows with NaNs are removed from the analysis.
        - 'raise': Raises an error if NaNs are present.
    random_state : int or np.random.Generator, optional
        Seed or random number generator for reproducibility of permutations.

    Returns
    -------
    PERMANOVAResult
        A dataclass containing the PERMANOVA results, including the pseudo-F statistic, p-value

    References
    ----------
    .. [1] Anderson, Marti J. "Permutational Multivariate Analysis of Variance (PERMANOVA)." Wiley StatsRef: Statistics Reference Online, Nov. 2017, pp. 1-15, https://doi.org/10.1002/9781118445112.stat07841. Accessed 5 Nov. 2026.
    .. [2] Anderson, Marti. (2017). Permutational Multivariate Analysis of Variance (PERMANOVA). 10.1002/9781118445112.stat07841.
    .. [3] "Effect Sizes for ANOVAs." Github.Io, 2018, https://easystats.github.io/effectsize/articles/anovaES.html. Accessed 11 Nov. 2026.
    """

    X = np.asarray(dependent, dtype=float)
    independent_array = np.asarray(independent)

    if X.ndim == 2 and independent_array.ndim == 2:
        group_list = [X, np.asarray(independent_array, dtype=float)]
        if any(group.shape[1] != group_list[0].shape[1] for group in group_list[1:]):
            raise ValueError("All groups must have the same number of features.")
        group_labels = np.concatenate(
            [np.full(group.shape[0], index) for index, group in enumerate(group_list)]
        )
        X = np.vstack(group_list)
    else:
        group_labels = independent_array
        if group_labels.ndim > 1:
            group_labels = group_labels.squeeze()
            if group_labels.ndim > 1:
                raise ValueError("Independent variable must be 1-dimensional.")
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError(
                "Dependent variable must be a 1D or 2D array-like of observations."
            )
        if group_labels.ndim != 1:
            raise ValueError("Independent variable must be 1-dimensional.")
        if X.shape[0] != group_labels.size:
            raise ValueError(
                "Dependent and independent arrays must have the same number of observations."
            )

    if X.ndim != 2:
        raise ValueError(
            "Dependent variable must be a 1D or 2D array-like of observations."
        )

    # handle nan as needed
    X, group_labels = _validate_nan_policy(X, group_labels, nan_policy)

    if np.isnan(X).any():
        return PERMANOVAResult(
            statistic=np.nan,
            pvalue=np.nan,
            eta_squared=np.nan,
            dof_between=np.nan,
            dof_within=np.nan,
        )

    n_samples = X.shape[0]
    unique_groups = np.unique(group_labels)
    if len(unique_groups) < 2:
        raise ValueError("PERMANOVA requires at least 2 distinct groups.")

    # compute distance matrix
    if distance_metric == "precomputed":
        # X is already a symmetric N x N distance matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError("Precomputed distance matrix must be square (N x N).")
        dist_sq = X**2
    else:
        # call to scipy pdist for easy computing
        condensed_dist = pdist(X, metric=distance_metric)
        dist_matrix = squareform(condensed_dist)
        dist_sq = dist_matrix**2

    # observed statistic calculation
    # compute the f-statistic for the current data
    f_obs, eta_sq_obs, _, dof_b, dof_w = compute_pseudo_f(
        dist_sq, group_labels, unique_groups, n_samples
    )

    # now run Monte Carlo Permutation Test
    rng = np.random.default_rng(random_state)
    perm_f_stats = np.zeros(permutations)  # to be set
    perm_labels = group_labels.copy()

    # we will determine the p-value through permutations
    # shuffle data labels between groups
    # compute the pseudo-F value for each permutation
    # store that in perm_f_stats
    for i in range(permutations):
        rng.shuffle(perm_labels)
        f_perm, _, _, _, _ = compute_pseudo_f(
            dist_sq, perm_labels, unique_groups, n_samples
        )
        perm_f_stats[i] = f_perm

    # p = (1 + number perm_f_stats > the observed F computed earlier) / (1 + total number of permutations)
    p_value = (np.sum(perm_f_stats >= f_obs) + 1.0) / (permutations + 1.0)

    return PERMANOVAResult(
        statistic=float(f_obs),
        pvalue=float(p_value),
        eta_squared=float(eta_sq_obs),
        dof_between=dof_b,
        dof_within=dof_w,
    )
