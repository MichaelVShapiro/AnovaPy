# handling N-Way ANOVAs
# this also includes alternative tests as needed

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np
from scipy.stats import f, rankdata, studentized_range, t

from anovapy.p_adjust import PValueAdjustmentMethod, adjust_pvalue

__all__ = ["art", "f_nway", "tukey_hsd"]


@dataclass
class NWayResult:
    """
    Result of `anovapy.nway.f_nway`.

    Attributes
    ----------
    effect : list[tuple[Any]]
        The effect being analyzed. Note that tuple at index i strictly depends on the effect being analyzed. For example, in a Three Way ANOVA, it could be just Factor A (for which ('A',) would be inputted) or the Three Way Interaction (which would be represented as ('A', 'B', 'C'))
    statistic : list[float]
        Computed F-statistic for each effect. Index i corresponds to index i in `effect` and `pvalue` here.
    pvalue : list[float]
        Computed p-value for each effect.
    df : list[int]
        Degrees of freedom for each effect
    df_error : int
        Degrees of freedom error (residual)
    """

    effect: np.ndarray
    statistic: np.ndarray
    pvalue: np.ndarray
    df: np.ndarray
    df_error: int


@dataclass
class ArtNWayResult:
    """
    Result of `anovapy.nway.art`.

    Attributes
    ----------
    effect : list[tuple[Any]]
        The effect being analyzed. Note that tuple at index i strictly depends on the effect being analyzed. For example, in a Three Way ANOVA, it could be just Factor A (for which ('A',) would be inputted) or the Three Way Interaction (which would be represented as ('A', 'B', 'C'))
    statistic : list[float]
        Computed F-statistic for each effect. Index i corresponds to index i in `effect` and `pvalue` here.
    pvalue : list[float]
        Computed p-value for each effect.
    df : np.ndarray
        Computed degrees of freedom for each effect
    df_error : int
        Degrees of Freedom Error (residual)
    """

    effect: list[tuple[str, ...]]
    statistic: list[float]
    pvalue: list[float]
    df: np.ndarray
    df_error: int


@dataclass
class PermutationNWayResult:
    """
    Result of `anovapy.nway.f_permutation`.

    Attributes
    ----------
    effect : list[tuple[Any]]
        The effect being analyzed. Note that tuple at index i strictly depends on the effect being analyzed. For example, in a Three Way ANOVA, it could be just Factor A (for which ('A',) would be inputted) or the Three Way Interaction (which would be represented as ('A', 'B', 'C'))
    statistic : list[float]
        Computed F-statistic for each effect. Index i corresponds to index i in `effect` and `pvalue` here.
    pvalue : list[float]
        Computed p-value for each effect.
    df : np.ndarray
        Computed degrees of freedom for each effect
    df_error : int
        Degrees of Freedom Error (residual)
    """

    effect: list[tuple[str, ...]]
    statistic: list[float]
    pvalue: list[float]
    df: np.ndarray
    df_error: np.ndarray


@dataclass
class TukeyResult:
    """
    Result of `anovapy.nway.tukey_hsd`.

    Each entry at index `i` across `comparisons`, `statistic`, and
    `pvalue` corresponds to a single pairwise comparison.

    Attributes
    ----------
    comparisons : list of tuple of tuple
        Pairwise comparisons formatted as `(group_a, group_b)`, where
        `group_a` and `group_b` are tuples of factor levels ordered
        according to `term`. For example, a 2-way interaction yields
        `((level_1a, level_2a), (level_1b, level_2b))`.
    statistic : np.ndarray
        1D array of computed Tukey Q-statistics corresponding to each
        entry in `comparisons`.
    pvalue : np.ndarray
        1D array of computed p-values corresponding to each entry in
        `comparisons`.
    """

    comparisons: list[tuple[Any, ...]]
    statistic: np.ndarray
    pvalue: np.ndarray


@dataclass
class ArtCResult:
    """
    Result of `anovapy.nway.artc`.

    Each entry at index `i` across `comparisons`, `statistic`, and
    `pvalue` corresponds to a single pairwise comparison.

    Attributes
    ----------
    comparisons : list of tuple of tuple
        Pairwise comparisons formatted as `(group_a, group_b)`, where
        `group_a` and `group_b` are tuples of factor levels ordered
        according to `term`. For example, a 2-way interaction yields
        `((level_1a, level_2a), (level_1b, level_2b))`.
    statistic : np.ndarray
        1D array of computed t-statistics corresponding to each
        entry in `comparisons`.
    pvalue : np.ndarray
        1D array of computed p-values corresponding to each entry in
        `comparisons`.
    """

    comparisons: list[tuple[Any, ...]]
    statistic: np.ndarray
    pvalue: np.ndarray


def _encode_factor(factor: np.ndarray) -> np.ndarray:
    """
    This is a helper function for `anovapy.nway.f_nway`.

    It One-hot encodes a 1D factor array and drops the first level (k - 1 columns).
    """
    _, inv = np.unique(factor, return_inverse=True)
    n_levels = inv.max() + 1
    # create (n_samples, n_levels - 1) dummy matrix
    dummies = np.eye(n_levels, dtype=float)[inv, 1:]
    return dummies


def _encode_factors(
    factor_indices: list[np.ndarray], shape: tuple[int, ...]
) -> np.ndarray:
    """
    Packs multiple 0-based integer factor arrays into a single flat cell index.
    """
    if len(factor_indices) == 1:
        return factor_indices[0]
    return np.ravel_multi_index(factor_indices, dims=shape)


def _compute_nway_type1(factors: list[np.ndarray], y: np.ndarray):
    """
    Computes Type I Sum of Squares for the given factor and dependent

    Parameters
    ----------
    factors : list[np.ndarray]
        All factors (independent variables). It is assumed there are at least 3
    y : np.ndarray
        The dependent variable.

    Returns
    -------
    An enormous tuple containing the effects analyzed, F-statistics, p-value, degrees of freedom, and degrees of freedom error (residual)
    """
    n_obs = len(y)
    n_factors = len(factors)

    # depending on how many factors there are, computationally this can get super expensive
    # we will optimize using Linear Model Projection (QR Decomposition)

    # first encode all the factors (main effects)
    main_blocks = [_encode_factor(f) for f in factors]

    # now build design matrix X sequentially (Intercept -> Main Effects -> Interactions)
    X_blocks = [np.ones((n_obs, 1))]  # intercept
    effect_slices = []
    col_idx = 1  # start after intercept

    # construct sequential blocks of design matrix X
    # depending on what 'order' we are at, it may represent single effect (i.e. just Factor A or Factor B) or different combinations (i.e. A:B) or just higher order interactions (i.e. A:B:C)
    for order in range(1, n_factors + 1):
        # generate and go through all possible combinations which we will analyze
        for i in combinations(range(n_factors), order):
            # start with the first factor in the effect
            # we will multiply everything in main_blocks[i] together essentially
            block = main_blocks[i[0]]  # to be changed
            # multiply row-wise with remaining factors in the interaction
            for f_idx in i[1:]:
                next_block = main_blocks[f_idx]
                # broadcasting to get all pairwise column products
                block = (block[:, :, None] * next_block[:, None, :]).reshape(n_obs, -1)

            X_blocks.append(block)
            n_cols = block.shape[1]
            # track which columns of X belong to this specific effect/interaction
            effect_slices.append((i, slice(col_idx, col_idx + n_cols)))
            col_idx += n_cols

    X = np.hstack(X_blocks)

    # now apply the QR Decomposition for Type I SS
    # this provides an orthonormal basis Q for the column space of X and avoids forming X.T @ X (which squares the condition number)
    Q, R = np.linalg.qr(X)
    # account for floating-point rounding error when checking numerical rank so we avoid empty or missing cells (the epsilon here is useful for that)
    tol = np.max(X.shape) * np.finfo(float).eps * np.abs(np.diag(R)).max()
    independent_cols = np.abs(np.diag(R)) > tol

    # project y onto the orthonormal columns of Q; squaring gives the variance contribution along each orthogonal basis vector
    proj_sq = (Q.T @ y) ** 2

    # now we can compute the actual sum of squares for each
    ss_total = np.sum(y**2)
    ss_model = np.sum(proj_sq[independent_cols])
    # compute sum of squares error (residual)
    ss_error = ss_total - ss_model
    # also compute degrees of freedom
    df_error = n_obs - np.sum(independent_cols)
    # compute mean squared error ahead of time
    ms_error = ss_error / df_error

    # extract results per effect
    effects, f_stats, p_vals, dfs = [], [], [], []
    for combo, slc in effect_slices:
        # get all columns belonging to this effect
        valid = independent_cols[slc]
        # now comes the easy part
        # just come the F-statistic and p-value like we usually do
        # compute degrees of freedom and sum of squares for the effect
        df_effect = np.sum(valid)
        ss_effect = np.sum(proj_sq[slc][valid])

        # compute mean squared effect
        ms_effect = ss_effect / df_effect
        # put it together
        f_val = ms_effect / ms_error
        p_val = f.sf(f_val, df_effect, df_error)

        effects.append(combo)
        f_stats.append(f_val)
        p_vals.append(p_val)
        dfs.append(df_effect)

    return tuple(effects), np.array(f_stats), np.array(p_vals), np.array(dfs), df_error


def _compute_only_mse(factors: list[np.ndarray], y: np.ndarray) -> tuple[float, int]:
    """
    This is an optimized function to compute the Mean Squared Error (MSE) and the Degrees of Freedom Error (residual).

    It is a helper function for `anovapy.nway.tukey_hsd`.

    The difference between this function and `anovapy.nway._compute_nway_type1` is that this one is explicitly optimized to **only** compute the MSE. Since Tukey's HSD only needs the MSE and no other Sum of Squares, we don't have to go through all the complexity of computing the MSE in `_compute_nway_type1`, we can compute the same MSE in O(n) time.

    Parameters
    ----------
    factors : list[np.ndarray]
        All independent variables
    y : np.ndarray
        The dependent variable

    Returns
    -------
    A tuple containing the computed MSE and Degrees of Freedom Error
    """
    # first get all unique items across all factors
    encoded = [np.unique(f, return_inverse=True)[1] for f in factors]
    combo_matrix = np.column_stack(encoded)
    unique_cells, cell_ids = np.unique(combo_matrix, axis=0, return_inverse=True)

    n_cells = len(unique_cells)
    n_obs = len(y)
    df_error = n_obs - n_cells

    # now compute cell means
    counts = np.bincount(cell_ids, minlength=n_cells)
    cell_means = np.bincount(cell_ids, weights=y, minlength=n_cells) / counts

    # now subtract each observation's cell mean from y and square
    residuals = y - cell_means[cell_ids]
    ss_error = np.dot(residuals, residuals)
    ms_error = ss_error / df_error

    return float(ms_error), int(df_error)


def _group_means_and_counts(
    factors: list[np.ndarray],
    y: np.ndarray,
    term: tuple[int, ...],
):
    """
    Another helper function for `anovapy.nway.tukey_hsd`.

    For each independent variable (and the selected term), this function computes the respective group mean.

    Parameters
    ----------
    factors : list[np.ndarray]
        All independent variables
    dependent : np.ndarray
        The dependent variable
    term : tuple[int, ...]
        The term being analyzed

    Returns
    -------
    The unique items in each factor, computed means, and counts
    """
    # first encode each selected factor into integer codes 0 .. L_k - 1
    unique_per_factor = [
        np.unique(factors[idx]) for idx in term
    ]  # this is important for the actually identifying each unique value directly
    encoded = [np.unique(factors[idx], return_inverse=True)[1] for idx in term]
    # now stack into shape (n_obs, len(term)) and find unique combinations (cells/levels)
    combo_matrix = np.column_stack(encoded)
    unique_combos, group_ids = np.unique(combo_matrix, axis=0, return_inverse=True)

    n_groups = len(unique_combos)
    counts = np.bincount(group_ids, minlength=n_groups)
    if np.any(counts == 0):
        raise ValueError(f"Empty cell encountered for term {term}.")

    sums = np.bincount(group_ids, weights=y, minlength=n_groups)
    means = sums / counts

    # make sure we can identify each label later
    level_labels: list[tuple[Any, ...]] = [
        tuple(unique_per_factor[col_idx][int(code)] for col_idx, code in enumerate(row))
        for row in unique_combos
    ]

    return unique_combos, level_labels, means, counts


def _main_effect_matrix(factor: np.ndarray) -> np.ndarray:
    """
    Treatment-coded dummy columns for one factor (dropping baseline level 0).

    This is a helper function for `anovapy.nway.f_permutation`.
    """
    _, inv = np.unique(factor, return_inverse=True)
    n_levels = int(inv.max()) + 1
    if n_levels <= 1:
        return np.empty((len(factor), 0))
    return inv[:, None] == np.arange(1, n_levels)


def _interaction_matrix(matrices: list[np.ndarray]) -> np.ndarray:
    """
    Row-wise Kronecker product across a list of 2D factor matrices.

    This is a helper function for `anovapy.nway.f_permutation`.
    """
    out = matrices[0]
    for mat in matrices[1:]:
        out = (out[:, :, None] * mat[:, None, :]).reshape(out.shape[0], -1)
    return out


def _build_comb_indices(
    factor_arrays: list[np.ndarray],
    cond_factors: list[int],
) -> list[np.ndarray]:
    """
    Return index arrays for each combination of levels defined by `cond_factors`.

    This is a helper function for `anovapy.nway.f_permutation`.
    """
    # check of conditional factors are empty
    # this happens for the highest order interactions (for example in a three way design this may be the interaction A:B:C)
    if not cond_factors:
        return []  # all cells get changed so nothing in specific to not change!

    cols = [factor_arrays[i] for i in cond_factors]
    dtype = [(f"f{k}", col.dtype, col.shape[1:]) for k, col in enumerate(cols)]
    packed = np.empty(len(cols[0]), dtype=dtype)
    for k, col in enumerate(cols):
        packed[f"f{k}"] = col

    # find unique combinations
    _, inv = np.unique(packed, return_inverse=True)
    # sort row indices so identical combinations sit next to each other
    order = np.argsort(inv)
    inv_sorted = inv[order]
    # split at boundaries where the group ID changes
    splits = np.flatnonzero(np.diff(inv_sorted)) + 1
    # keeps only groups with more than 1 observation since a group of size 1 has nothing to permute
    return [idx for idx in np.split(order, splits) if len(idx) > 1]


def f_nway(
    *factors: np.typing.ArrayLike,
    dependent: np.typing.ArrayLike,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> NWayResult:
    """
    Runs an N-Way ANOVA.

    The `N` here is determined by the number of factors passed in. For example, when three factors are passed in, a Three Way ANOVA is run.

    Parameters
    ----------
    *factors : array_like
        All factors to pass in. Each factor should be a 1D array. A minimum of three factors must be passed in
    dependent : array_like
        The dependent variable. This should be a 1D array
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle NaN inputs

    Returns
    -------
    NWayResult
        A dataclass containing F-Statistics and p-values computed

    See Also
    --------
    scipy.stats.f_oneway : One Way ANOVA
    anovapy.twoway.f_twoway : Two Way ANOVA

    Raises
    ------
    `ValueError` When less than three factors are passed in.

    Notes
    -----
    This function, at least as of v0.2 of AnovaPy, only implemented Type I Sum of Squares.

    At least three factors (independent variables) should be passed in. Passing in two, to perform a Two Way ANOVA, will result in an error.
    Use `anovapy.twoway.f_twoway` which is optimized to handle a Two Way ANOVA.
    For handling One Way designs (i.e. One Way ANOVA) use `scipy.stats.f_oneway`.

    References
    ----------
    """
    factors = np.asarray(factors)
    dependent = np.asarray(dependent)

    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # handle lengths as well
    if len(factors) == 2:
        raise ValueError(
            "At least three factors are needed but only 2 were given; use anovapy.twoway.f_twoway to handle Two Way designs."
        )
    if len(factors) < 3:
        raise ValueError("At least three factors are needed to run an N-Way ANOVA.")

    # handle nan
    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "propagate":
            return NWayResult(
                effect=np.array(np.nan),
                statistic=np.array(np.nan),
                pvalue=np.array(np.nan),
                df=np.array(np.nan),
                df_error=np.nan,
            )
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        # omit
        dependent = dependent[~np.isnan(dependent)]
        # omit from factors too
        mask = np.isnan(dependent)
        for i in range(len(factors)):
            factors[i] = factors[i][~mask]

    # compute Type I Sum of Squares
    effects, stats, p_vals, dfs, err = _compute_nway_type1(factors, dependent)

    return NWayResult(
        effect=effects, statistic=stats, pvalue=p_vals, df=dfs, df_error=err
    )


def tukey_hsd(
    *factors: np.typing.ArrayLike,
    dependent: np.typing.ArrayLike,
    term: tuple[int, ...] = (0,),
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> TukeyResult:
    """
    Performs Tukey's HSD for N-Way designs.

    Parameters
    ----------
    *factors : array_like
        Factor arrays representing the independent variables. Their positional
        order defines the 0-based indices used in `term`.
    dependent : array_like
        The dependent variable
    term : tuple
        Non-empty tuple of 0-based factor indices specifying which main effect
        or interaction to test. For example, `(0,)` tests the first factor,
        `(0, 2)` tests the interaction between the first and third factors,
        and `(0, 1, 2)` tests the three-way interaction. Defaults to `(0,)`.
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle inputs with NaN values.

    Returns
    -------
    TukeyResult
        Dataclass containing `comparisons`, `statistic` (Q-statistics),
        and `pvalue` for each tested pair.

    Raises
    ------
    ValueError
        If fewer than three factors are provided.

    Notes
    -----
    If needed, this function will automatically apply the Tukey-Kramer adjustment.

    When comparing against R's `TukeyHSD`, test statistics match closely,
    but very small p-values (e.g. on the order of `1e-12`) may show minor
    differences due to numerical integration differences between SciPy's
    `studentized_range` (see `scipy.stats.studentized_range`) and R's internal studentized range calculation.
    Practical conclusions and relative ordering of comparisons remain consistent.

    References
    ----------
    .. [1] Tukey, John W. "Comparing Individual Means in the Analysis of Variance." Biometrics, vol. 5, no. 2, JSTOR, June 1949, p. 99, https://doi.org/10.2307/3001913. Accessed 11 Nov. 2026.
    """
    factors = np.asarray(factors)
    dependent = np.asarray(dependent)

    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    if len(factors) == 2:
        raise ValueError(
            "At least three factors are needed but only two were passed in; for two factors, make a call to `anovapy.twoway.tukey_hsd` which is optimized for Two Way Designs."
        )
    elif len(factors) < 3:
        raise ValueError(
            "At least three factors are needed to compute Tukey's HSD for N-Way designs."
        )

    if len(term) == 0:
        raise ValueError(
            "Cannot compute Tukey's HSD because no term is being analyzed!"
        )

    # handle nan
    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "propagate":
            return TukeyResult(
                comparisons=[], statistic=np.array(np.nan), pvalue=np.array(np.nan)
            )
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        # omit
        mask = np.isnan(dependent)
        dependent = dependent[~mask]
        for i in range(len(factors)):
            factors[i] = factors[i][~mask]

    # the overall logic is the same as computing Tukey HSD with One Way or Two Way designs
    # get all items we are trying to analyze (which we will compute the mean of)
    # and compute the MS Error
    # computing that requires computing sum of squares error (residual) and degrees of freedom error
    # once again though, the same problem arises -- complexity
    # depending on how many terms are being analyzed, this can explode
    # let's first compute the mean squared error (MSE)
    mse, df_err = _compute_only_mse(factors, dependent)

    # now get group labels, means, and counts for the specific `term` in O(n) time
    unique_levels, level_labels, means, counts = _group_means_and_counts(
        factors, dependent, term
    )
    n_groups = len(unique_levels)

    # apply the Tukey-Kramer adjustment if lengths between comparisons are not the same
    pairs = np.array(
        list(combinations(range(n_groups), 2)), dtype=int
    )  # generate all possible effects to analyze as needed
    idx_a, idx_b = pairs[:, 0], pairs[:, 1]

    n_a, n_b = counts[idx_a], counts[idx_b]
    mean_diff = np.abs(means[idx_a] - means[idx_b])
    se = np.sqrt(mse * 0.5 * (1.0 / n_a + 1.0 / n_b))
    q_stats = mean_diff / se
    p_values = studentized_range.sf(q_stats, n_groups, df_err)

    # now we just need to map back idx_a and idx_b back to the original level names
    # that way in the return result we know exactly what each statistic and p-value corresponds to!
    comparisons: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = [
        (level_labels[int(a)], level_labels[int(b)]) for a, b in zip(idx_a, idx_b)
    ]

    return TukeyResult(comparisons=comparisons, statistic=q_stats, pvalue=p_values)


def art(
    *factors: np.typing.ArrayLike,
    dependent: np.typing.ArrayLike,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> ArtNWayResult:
    """
    Performs the Aligned Rank Transform (ART) N-Way ANOVA.

    This can be used when assumptions for N-Way ANOVA are not met.

    Parameters
    ----------
    *factors : array_like
        All factors to pass in. Each factor should be a 1D array. A minimum of three factors must be passed in
    dependent : array_like
        The dependent variable. This should be a 1D array
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle NaN inputs

    Returns
    -------
    ArtNWayResult
        A dataclass containing F-Statistics and p-values computed

    See Also
    --------
    anovapy.twoway.art : ART Two Way ANOVA
    anovapy.oneway.art : ART One Way ANOVA

    Notes
    -----
    To start with, this function is not to be confused with `anovapy.twoway.art`. Note that while both *are* ART ANOVA, they are specialized for different inputs; `anovapy.twoway.art` is ART Two Way ANOVA, and has been built in a way where it is specifically optimized for handling Two Way designs, as opposed to this function which is designed to handle N-Way designs.
    Hence trying to pass in only two factors here will result in an error.

    Contrary to `anovapy.twoway.art` which uses Type III Sum of Squares by default, this function uses Type I at the moment.

    References
    ----------
    .. [1] Wobbrock, J. O., Findlater, L., Gergle, D., & Higgins, J. J. (2011). The aligned rank transform for nonparametric factorial analyses using only ANOVA procedures. Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, 143-146. https://doi.org/10.1145/1978942.1978963
    """
    factors = [np.asarray(i) for i in factors]
    dependent = np.asarray(dependent)

    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # validate lengths
    if len(factors) == 2:
        raise ValueError(
            "At least three factors are needed but only 2 were given; use anovapy.twoway.art to handle Two Way designs."
        )
    if len(factors) < 3:
        raise ValueError("At least three factors are needed to run an N-Way ART ANOVA.")

    # handle nan
    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "propagate":
            return ArtNWayResult(
                effect=np.array(np.nan),
                statistic=np.array(np.nan),
                pvalue=np.array(np.nan),
                df=np.array(np.nan),
                df_error=np.nan,
            )
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        # omit
        dependent = dependent[~np.isnan(dependent)]

    # the operation here isn't that different from anovapy.twoway.art
    # we are just doing the same thing but with more factors involved!

    # so first compute the residuals
    # compute cell means
    keys = np.column_stack(factors)
    _unique_cells, inverse = np.unique(keys, axis=0, return_inverse=True)

    # compute sums and counts per group
    sums = np.bincount(inverse, weights=dependent)
    counts = np.bincount(inverse)

    # now compute the cell mean by simply dividing the two
    cell_means_unique = sums / counts

    # now just map out the indecies so we can easily compute residuals
    cell_means = cell_means_unique[inverse]

    residuals = dependent - cell_means

    # compute estimated effects
    grand_mean = np.mean(dependent)

    # compute effects
    # some parts remain similar to that of anovapy.twoway.art
    # for instance a one way effect will always be: A_i - grand_mean
    # and the interaction will always be: a_i*b_i - a_i - b_i + grand_mean
    # for three way effects (i.e. three way anova), the formula is: A_i*B_i*C_i - A_i*B_i - A_i*C_i - B_i*C_i + A_i + B_i + C_i - grand_mean
    # for four way effects (i.e. four way anova), the formula is: A_i*B_i*C_i*D_i - A_i*B_i*C_i - A_i*B_i*D_i - A_i*C_i*D_i - B_i*C_i*D_i + A_i*B_i + A_i*C_i + A_i*D_i + B_i*C_i + B_i*D_i + C_i*D_i - A_i - B_i - C_i - D_i + grand_mean
    # so there really is a pattern here!
    # for example, let h represent the *n* in N-Way ANOVA (for example, in a three way ANOAV h is simply 3, in 4 Way ANOVA it is 4, and so on); to put it simply, assume *h* refers to how many factors we have
    # where h is even, we add the grand mean at the end, otherwise we subtract
    # so multiply all effects together to find the proper mean, and go back and forth between subtracting and adding depending what level you are at just like the formula given in the paper:
    # N-Way Effects: NWay - sum((N-1 Way)) + sum((N - 2 Way)) - sum((N - 3 Way)) + sum((M - 4 Way)) ... +- sum((N-h Way))
    # note that as N increases the sign (+/-) also changes in an "inclusion-exclusion" like way

    # this is where it gets tricky though -- computing means literally for every possible combination!
    # time complexity here can blow up
    # thankfully we might be able to optimize somehow
    # since we are going to be making many calls to getting cell means
    # store it all in a cache (dictionary) which we will make many calls to over time
    cell_mean_cache = {
        (): np.full_like(dependent, grand_mean)
    }  # the key here is a tuple of factor indecies and the value represents the cell means mapped to each observation

    # a helper function for computing cell (effect) means
    def get_cell_means(factor_indices):
        if not factor_indices:
            # empty set represents the grand mean
            return np.full_like(dependent, grand_mean)

        # form a unique key based on the given indecies
        key = tuple(factor_indices)
        # check if mean was already computed
        # this saves a bunch of time
        if key in cell_mean_cache:
            return cell_mean_cache[key]  # perfect!

        # compute mean of cell
        # we can use the same technique as earlier just on the given factor indecies only!
        keys = np.column_stack([factors[i] for i in factor_indices])
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        sums = np.bincount(inverse, weights=dependent)
        counts = np.bincount(inverse)

        cell_means = (sums / counts)[inverse]
        cell_mean_cache[key] = cell_means
        return cell_means

    # now generate all possible factor combinations
    # this will vary depending on what N (as in N-Way ANOVA) really is
    # for example, if N = 3, we should have possible combinations such as: (0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)
    # lets follow that format so generate all possible combinations in standard order
    N = len(factors)
    effect_list = [comb for k in range(1, N + 1) for comb in combinations(range(N), k)]

    # now compute estimated effects for all main and interaction effects
    # really just follow the formula
    num_effects = len(effect_list)
    effect_res = np.empty(num_effects, dtype=object)
    stats_res = np.empty(num_effects)
    pval_res = np.empty(num_effects)
    dfs = np.empty(num_effects)
    df_err = 0

    # now go through each effect, and align, rank and run a factorial ANOVA
    for idx, s_indices in enumerate(effect_list):
        k = len(s_indices)
        effect_values = np.zeros_like(dependent, dtype=float)

        # now apply the "inclusion-exclusion" sum for estimated effect
        # basically as stated in the formula, the sign changes depending on what "N" is per iteration
        for r in range(k + 1):
            sign = (-1) ** (k - r)
            for t_indices in combinations(s_indices, r):
                effect_values += sign * get_cell_means(t_indices)

        # now align and rank: Align (Y') and Rank (Y'')
        aligned = residuals + effect_values
        ranked = rankdata(aligned, method="average")

        # now we can run the full factorial anova on the aligned and ranked data
        temp = f_nway(*factors, dependent=ranked)

        effect_res[idx] = temp.effect[idx]
        stats_res[idx] = temp.statistic[idx]
        pval_res[idx] = temp.pvalue[idx]
        dfs[idx] = temp.df[idx]
        df_err = (
            temp.df_error
        )  # degrees of freedom error (residual) should be the same regardless

    return ArtNWayResult(
        effect=effect_res, statistic=stats_res, pvalue=pval_res, df=dfs, df_error=df_err
    )


def artc(
    *factors: np.typing.ArrayLike,
    dependent: np.typing.ArrayLike,
    term: tuple[int, ...] = (0,),
    adjust: PValueAdjustmentMethod = "bonferroni",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> ArtCResult:
    """
    Performs the Aligned Rank Transform - Contrast (ART-C) post-hoc test to ART ANOVA.

    Parameters
    ----------
    *factors : array_like
        Factor arrays representing the independent variables. Their positional
        order defines the 0-based indices used in `term`.
    dependent : array_like
        The dependent variable
    term : tuple
        Non-empty tuple of 0-based factor indices specifying which main effect
        or interaction to test. For example, `(0,)` tests the first factor,
        `(0, 2)` tests the interaction between the first and third factors,
        and `(0, 1, 2)` tests the three-way interaction. Defaults to `(0,)`.
    adjust : PAdjustmentMethod
        Defines how to adjust p-values. Defaults to 'bonferroni'.
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle inputs with NaN values.

    Returns
    -------
    ArtCResult
        Dataclass containing `comparisons`, `statistic` (t-statistics),
        and `pvalue` for each tested pair.

    Raises
    ------
    ValueError
        If fewer than three factors are provided.

    Notes
    -----
    This implementation follows the alignment and contrast procedure described in Elkin et al. (2021) directly in IEEE-754 float64 arithmetic without intermediate rounding prior to ranking. Consequently, minor numerical differences (typically beyond the second or third decimal place) may occur relative to other implementations (such as ARTool) when algebraically tied aligned values differ at floating-point machine epsilon ($\approx 10^{-16}$) and are assigned distinct rather than averaged ranks.

    References
    ----------
    .. [1] Elkin, L. A., Kay, M., Higgins, J. J., & Wobbrock, J. O. (2021). An aligned rank transform procedure for multifactor contrast tests. The 34th Annual ACM Symposium on User Interface Software and Technology. https://doi.org/10.1145/3472749.3474784
    """
    factors = np.asarray(factors)
    dependent = np.asarray(dependent)

    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    if len(factors) == 2:
        raise ValueError(
            "At least three factors are needed but only two were passed in; for two factors, make a call to `anovapy.twoway.artc` which is optimized for Two Way Designs."
        )
    elif len(factors) < 3:
        raise ValueError(
            "At least three factors are needed to compute Tukey's HSD for N-Way designs."
        )

    if len(term) == 0:
        raise ValueError("Cannot compute ART-C because no term is being analyzed!")

    # now handle nan
    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "propagate":
            return ArtCResult(
                comparisons=[], statistic=np.array(np.nan), pvalue=np.array(np.nan)
            )
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")

        # omit
        mask = np.isnan(dependent)
        dependent = dependent[~mask]
        for i in range(len(factors)):
            factors[i] = factors[i][~mask]

    # first concatenate factors
    # this can be tricky though as depending on what 'N' is, there could be so many possible combinations!
    # in fact, what we are analyzing is equally important
    # contrary to anovapy.twoway.artc, we will try to concatenate all factors using indecies
    n_obs = len(dependent)
    n_factors = len(factors)
    levels_list = []
    codes_list = []
    for i in factors:
        uniques, codes = np.unique(i, return_inverse=True)
        levels_list.append(uniques)
        codes_list.append(codes)

    shape_all = tuple(len(u) for u in levels_list)
    full_cell_id = _encode_factors(codes_list, shape_all)

    # now separate factor indices into concatenated (term) and remaining (rem)
    term_idx = list(term)
    rem_idx = [i for i in range(n_factors) if i not in term_idx]

    shape_term = tuple(shape_all[i] for i in term_idx)
    term_id = _encode_factors([codes_list[i] for i in term_idx], shape_term)
    n_term_levels = int(np.prod(shape_term))

    # now we align and rank
    # to compute aligned response Y', it depends on how many factors we are dealing with
    # for three factor the formula is: Y_ijk' = Y_ijk - ABC_mean + ABC_mean - grand_mean
    # for two factor it is: Y_ijk' = Y_ijk - AB_meanC_mean + AB_mean - grand_mean
    # for single factor it is: Y_ijk' = Y_ijk - ABC_mean + A_mean - grand_mean
    # for higher levels, the formula really just follows: Y_ijk' = Y_ijk - factors_N_mean + factors_N_mean - grand_mean
    grand_mean = np.mean(dependent)
    # compute the ABC_mean part of the formula (just the full cell means that's all)
    full_counts = np.bincount(full_cell_id, minlength=int(np.prod(shape_all)))
    full_means = (
        np.bincount(full_cell_id, weights=dependent, minlength=len(full_counts))
        / np.maximum(full_counts, 1)
    )[full_cell_id]

    # now compute the means for the specific term (like AB_mean or A_mean or factors_N_mean dependent on what term is selected)
    term_counts = np.bincount(term_id, minlength=n_term_levels)
    term_means = (
        np.bincount(term_id, weights=dependent, minlength=n_term_levels)
        / np.maximum(term_counts, 1)
    )[term_id]

    # put it all together to compute Y'
    y_prime = dependent - full_means + term_means - grand_mean
    # now compute ranked response Y''
    y_double_prime = rankdata(y_prime, method="average")

    # now we conduct contrast tests on Y''

    # compute pooled MSE from full-factorial cell residuals
    cell_sums = np.bincount(
        full_cell_id, weights=y_double_prime, minlength=len(full_counts)
    )
    non_empty = full_counts > 0
    cell_means = np.zeros_like(cell_sums)
    cell_means[non_empty] = cell_sums[non_empty] / full_counts[non_empty]

    # compute sum of squares error (residual)
    sse = np.sum((y_double_prime - cell_means[full_cell_id]) ** 2)
    df_res = n_obs - int(np.sum(non_empty))  # degrees of freedom error (residual)
    mse = sse / df_res  # mean squared error

    # reshape cell means & counts into a (n_term_levels, J) grid for marginal mean contrasts
    if rem_idx:
        shape_rem = tuple(shape_all[i] for i in rem_idx)
        rem_id = _encode_factors([codes_list[i] for i in rem_idx], shape_rem)
        n_rem_levels = int(np.prod(shape_rem))
    else:
        rem_id = np.zeros(n_obs, dtype=np.intp)
        n_rem_levels = 1

    # compute the means of each cell/effect (estimated marginal means)
    # we need this to compute the t-statistic
    grid_id = term_id * n_rem_levels + rem_id
    grid_counts = np.bincount(grid_id, minlength=n_term_levels * n_rem_levels).reshape(
        n_term_levels, n_rem_levels
    )
    grid_sums = np.bincount(
        grid_id, weights=y_double_prime, minlength=n_term_levels * n_rem_levels
    ).reshape(n_term_levels, n_rem_levels)

    # check for missing cells (empty combinations in unbalanced designs)
    if np.any(grid_counts == 0):
        raise ValueError(
            "Design contains empty cells; unbalanced designs with missing cells require pseudoinverse Estimated Marginal Means weights."
        )

    grid_means = grid_sums / grid_counts
    # now compute marginal means
    emms = np.mean(grid_means, axis=1)  # shape: (n_term_levels,)
    inv_counts_sum = np.sum(1.0 / grid_counts, axis=1)  # shape: (n_term_levels,)

    # now vectorized pairwise contrasts across all pairs (i, j)
    valid_levels = np.arange(n_term_levels)
    # get all possible combinations we are evaluating
    pairs = np.array(list(combinations(valid_levels, 2)))
    idx_a, idx_b = pairs[:, 0], pairs[:, 1]

    # compute estimate
    estimates = emms[idx_a] - emms[idx_b]
    # compute standard error
    ses = np.sqrt(
        mse * (inv_counts_sum[idx_a] + inv_counts_sum[idx_b]) / (n_rem_levels**2)
    )
    # put it together to compute the t-statistic
    t_stats = estimates / ses
    # compute p-value
    p_vals = 2.0 * t.sf(np.abs(t_stats), df_res)

    # map integer term IDs back to human readable tuples
    unraveled = np.unravel_index(valid_levels, shape_term)
    level_tuples = [
        tuple(levels_list[term_idx[k]][unraveled[k][lvl]] for k in range(len(term_idx)))
        for lvl in valid_levels
    ]
    comparisons = [(level_tuples[a], level_tuples[b]) for a, b in pairs]

    # finally adjust p-values
    p_vals = adjust_pvalue(p_vals, adjust)

    return ArtCResult(comparisons=comparisons, statistic=t_stats, pvalue=p_vals)


def f_permutation(
    *factors: np.typing.ArrayLike,
    dependent: np.typing.ArrayLike,
    method: Literal["manly", "edgington", "braak"] = "manly",
    permutations: int = 999,
    random_state: int | np.random.Generator | None = None,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> PermutationNWayResult:
    """
    Runs a Permutation N-Way ANOVA.

    This can be used when assumptions of ANOVA are not met.

    Parameters
    ----------
    *factors : array_like
        All factors to pass in. Each factor should be a 1D array. A minimum of three factors must be passed in
    dependent : array_like
        The dependent variable. This should be a 1D array
    method : {'manly', 'edgington', 'braak'}
        The permutation method to use. Options are:

        - 'manly': Use Manly's method to permute the data
        - 'edgington': Use Edgington's method to permute the data
        - 'braak': Use ter Braak's method to permute the data

        Defaults to 'manly'.
    permutations : int
        Number of permutations to run. Defaults to 999
    random_state : int | np.random.Generator | None
        A seed to set for reproducible results
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle inputs with NaN values.

    Returns
    -------
    PermutationNWayResult
        A dataclass containing computed statistics, p-values, and degrees of freedom.

    References
    ----------
    .. [1] Anderson, Marti J. "Permutation Tests for Univariate or Multivariate Analysis of Variance and Regression." Canadian Journal of Fisheries and Aquatic Sciences, vol. 58, no. 3, Mar. 2001, pp. 626-39, https://doi.org/10.1139/f01-004.
    .. [2] Anderson, Marti, and Cajo Ter Braak. "Permutation Tests for Multi-Factorial Analysis of Variance." Journal of Statistical Computation and Simulation, vol. 73, no. 2, Jan. 2003, pp. 85-113, https://doi.org/10.1080/00949650215733.
    .. [3] Edgington, Eugene S. Randomization Tests. 3rd ed., Marcel Dekker, 1995.
    """
    factors = np.asarray(factors)
    dependent = np.asarray(dependent)

    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")
    if method not in {"manly", "edgington", "braak"}:
        raise ValueError("method must be one of {'manly', 'edgington', 'braak'}.")

    # handle nan
    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "propagate":
            return PermutationNWayResult(
                effect=[],
                statistic=np.array(np.nan),
                pvalue=np.array(np.nan),
                df=np.array(np.nan),
                df_error=np.array(np.nan),
            )
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")

        # omit
        mask = np.isnan(dependent)
        dependent = dependent[~mask]
        for i in range(len(factors)):
            factors[i] = factors[i][~mask]

    # first get the observed statistics
    F_obs = f_nway(*factors, dependent=dependent)
    num_greater = np.zeros(F_obs.statistic.shape)  # to be filled

    n_obs = len(dependent)
    n_factors = len(factors)

    # extract effect tuples
    effect_tuples: list[tuple[int, ...]] = [
        tuple(int(idx) for idx in eff) for eff in F_obs.effect
    ]

    gen = np.random.default_rng(random_state)

    # make a local copy to keep the original safe
    y = dependent.copy()

    def fit_ols(design: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return OLS fitted values `X @ beta`.
        """
        if design.shape[1] == 0:
            return np.zeros_like(y)
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        return design @ beta

    # depending on what method is selected, precompute required data
    if method == "edgington":
        # with edgington, we test very specific terms/effects
        # so permute data a part of the specified term/label without changing other term/labels
        # so we will make conditional factors where for a given effect (could be just a single factor or an interaction), we will store a list of all other factor indices
        # we will also store a specific combination of levels across conditioning factors
        comb_by_effect: list[list[np.ndarray]] = []
        for term_factors in effect_tuples:
            # make the conditional factors that will define the blocks/subgroups we will condition on
            cond_factors = [i for i in range(n_factors) if i not in term_factors]
            comb_by_effect.append(_build_comb_indices(factors, cond_factors))
    elif method == "braak":
        # in ter Braak we test a specific term by fitting the full model (all main effects and interactions) to get residuals
        # then we fit a reduced model (everything except the term we are permuting) to get baseline predictions without that effect, and adding shuffled full model residuals onto those reduced predictions
        # first convert each factor array into a numeric treatment coded (dummy) indicator matrix
        main_mats = [_main_effect_matrix(f) for f in factors]
        all_terms: list[tuple[int, ...]] = []
        # generate every possible ANOVA term across orders 1 through n_factors
        for order in range(1, n_factors + 1):
            all_terms.extend(combinations(range(n_factors), order))

        # lets make a simple dict where we map each term tuple to its column block in the design matrix
        term_blocks: dict[tuple[int, ...], np.ndarray] = {}
        for term in all_terms:
            if len(term) == 1:
                term_blocks[term] = main_mats[term[0]]  # just a single factor!
            else:
                # produce all interaction dummy columns for each observation
                term_blocks[term] = _interaction_matrix([main_mats[i] for i in term])

        intercept = np.ones((n_obs, 1), dtype=float)
        X_full = np.hstack([intercept] + [term_blocks[t] for t in all_terms])
        residuals = y - fit_ols(X_full, y)

        reduced_fits: dict[tuple[int, ...], np.ndarray] = {}
        for term_factors in effect_tuples:
            other_blocks = [term_blocks[t] for t in all_terms if t != term_factors]
            X_reduced = (
                np.hstack([intercept] + other_blocks) if other_blocks else intercept
            )
            reduced_fits[term_factors] = fit_ols(X_reduced, y)

    # now permute depending on what method we are dealing with
    for _ in range(permutations):
        if method == "manly":
            # just shuffle dependent only
            gen.shuffle(y)
            # now run N-Way ANOVA
            temp = f_nway(*factors, dependent=y)

            # compare and adjust
            num_greater += np.asarray(temp.statistic) >= F_obs.statistic
        elif method == "edgington":
            # go through each possible effect and shuffle data only in those cells
            for eff_idx, term_factors in enumerate(effect_tuples):
                comb = comb_by_effect[eff_idx]
                if not comb:
                    # this must be the highest order interaction
                    # permute exactly like the manly method
                    y_perm = y.copy()
                    gen.shuffle(y_perm)
                else:
                    # make a local copy to keep the original safe
                    y_perm = y.copy()
                    # shuffle the row indices inside this stratum
                    # then pull those y values in a permuted order and write them back into their positions
                    for idx in comb:
                        y_perm[idx] = y[gen.permutation(idx)]

                # run N-Way ANOVA and compare statistics
                temp = f_nway(*factors, dependent=y_perm)
                # extract only the statistic for the effect we are analyzing
                stat_val = temp.statistic[eff_idx]
                # compare and adjust as needed
                if stat_val >= F_obs.statistic[eff_idx]:
                    num_greater[eff_idx] += 1
        else:  # ter Braak
            # make a local copy to keep the original safe
            permuted_res = residuals.copy()
            # shuffle the full model residuals once for this permutation draw
            gen.shuffle(permuted_res)

            for eff_idx, term_factors in enumerate(effect_tuples):
                # construct a simulated response under the reduced model (this would be without term_factors) and the shuffled residuals from the full model
                y_sim = reduced_fits[term_factors] + permuted_res
                # run N-Way ANOVA
                temp = f_nway(*factors, dependent=y_sim)
                # get statistic for the effect being tested
                stat_val = temp.statistic[eff_idx]
                # compare and adjust as needed
                if stat_val >= F_obs.statistic[eff_idx]:
                    num_greater[eff_idx] += 1

    # now compute p-value
    p_vals = (num_greater + 1) / (permutations + 1)

    return PermutationNWayResult(
        effect=F_obs.effect,
        statistic=F_obs.statistic,
        pvalue=p_vals,
        df=F_obs.df,
        df_error=F_obs.df_error,
    )
