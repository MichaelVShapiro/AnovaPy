# handling N-Way ANOVAs
# this also includes alternative tests as needed

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

import numpy as np
from scipy.stats import f, rankdata, studentized_range

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
