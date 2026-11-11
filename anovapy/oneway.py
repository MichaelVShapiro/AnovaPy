# everything here is purly scipy just with its own functions
# and naming conventions for simplicity
# a few more functions though are added

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import helmert
from scipy.stats import f, norm, rankdata, ttest_rel

__all__ = ["brown", "dunns", "f_repeated", "pairwise_ttest"]

BrownResult = namedtuple("BrownResult", ["statistic", "pvalue"])
RepeatedMeasuresAnovaResult = namedtuple(
    "RepeatedMeasuresAnovaResult", ["statistic", "pvalue"]
)


@dataclass
class DunnsResult:
    """
    Result of `anovapy.oneway.dunns`.

    Attributes
    ----------
    comparisons : list[tuple[int, int]]
        All comparisons being made.
    statistic : np.ndarray
        All computed Z-statistics
    pvalue : np.ndarray
        All computed p-values
    """

    comparisons: list[tuple[int, int]]
    statistic: np.ndarray
    pvalue: np.ndarray


@dataclass
class AdjustedTTestResult:
    """
    Result of `anovapy.oneway.adjusted_ttest`.

    Attributes
    ----------
    statistic : np.ndarray
        Computed t-statistic for each comparison
    pvalue : np.ndarray
        Computed p-value
    """

    statistic: np.ndarray
    pvalue: np.ndarray


def dunns(
    *samples: np.typing.ArrayLike,
    adjust: Literal["bonferroni", "holm", "sidak", "benjamini", "none"] = "bonferroni",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> DunnsResult:
    """
    Does Dunn's Post-Hoc Test. This is typically used after running a Kruskal-Wallis test (see `scipy.stats.kruskal`).

    Parameters
    ----------
    *samples : array_like
        All observations
    adjust : {'bonferroni', 'holm', 'sidak', 'benjamini', 'none'}
        The p-value adjustment to use. Defaults to 'bonferroni'
        - 'bonferroni' : Use the Bonferroni adjustment method
        - 'holm' : Use the Holm-Bonferroni adjustment method
        - 'sidak' : Use the Sidak adjustment method
        - 'benjamini' : Use the Benjamini-Hochberg adjustment method
        - 'none' : Do not adjust the p-value
    nan_policy : {'propagate', 'omit', 'raise'}
        The nan policy

    Returns
    -------
    DunnsResult
        A dataclass containing:
        - `comparisons` : A tuple that contains two numbers representing the index of the groups being compared. Note that this index is relative to how the groups were passed into the function.
        - `statistic` : Computed Z-statistics for each comparison
        - `p_value_unadjusted` : Unadjusted p-value
        - `p_value_adjusted` : Adjusted p-value (based on what was set in `adjust`)

    References
    ----------
    .. [1] Statistics Fundamentals. "Kruskal-Wallis Test: Formula, Examples, Calculator & Interpretation." Statisticsfundamentals.Com, Statistics Fundamentals, 16 Aug. 2026, https://statisticsfundamentals.com/hypothesis-testing/kruskal-wallis-test/. Accessed 11 Nov. 2026.
    .. [2] Bobbitt, Zach. "Dunn's Test for Multiple Comparisons." Statology, 29 Sept. 2020, https://www.statology.org/dunns-test/. Accessed 11 Nov. 2026.
    """
    samples = [np.asarray(i) for i in samples]

    samples = np.array(samples)

    # handle nan input
    has_nan = np.isnan(samples).any()
    if has_nan:
        if nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        if nan_policy == "propagate":
            return DunnsResult(
                statistic=np.array(np.nan),
                p_value_unadjusted=np.array(np.nan),
                p_value_adjusted=np.array(np.nan),
            )
        samples = samples[~np.isnan(samples)]

    n, p = samples.shape

    N = n * p

    res_stats = []
    res_pvals_unadjusted = []
    res_pvals_adjusted = []
    res_groups = []

    # compute means ahead of time for each group
    # rank all observations together
    all_data = samples.flatten()
    ranks = rankdata(all_data)

    # split ranks back into groups and compute mean ranks
    ranks = ranks.reshape(n, p)
    group_means = np.mean(ranks, axis=1)

    # compute tie correction ahead of time as well
    _, counts = np.unique(samples, return_counts=True)
    # get counts that appear > 1
    tie_sizes = counts[counts > 1]
    C = 1 - np.sum(tie_sizes**3 - tie_sizes) / (N**3 - N)

    # go through each group
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            res_groups.append((i, j))
            # z_{ij} = (R_iu - R_ju) / SE_ij
            numerator = abs(group_means[i] - group_means[j])

            # standard error = sqrt((N(N + 1) / 12) * (1/n_i + 1/n_j) * C)
            pt1 = N * (N + 1) / 12
            pt2 = 1 / p + 1 / p
            std_err = np.sqrt(pt1 * pt2 * C)
            z_stat = numerator / std_err

            res_stats.append(z_stat)

            # compute unadjusted p-value
            p_unadj = 2 * norm.sf(z_stat)
            res_pvals_unadjusted.append(p_unadj)

    # now adjust all p-values
    m = len(res_pvals_unadjusted)
    res_pvals_unadjusted = np.array(res_pvals_unadjusted)
    if adjust == "bonferroni":
        for k in res_pvals_unadjusted:
            res_pvals_adjusted.append(min(1.0, k * m))
    elif adjust == "holm":
        order = np.argsort(res_pvals_unadjusted)
        sorted_p = res_pvals_unadjusted[order]

        adjusted_sorted = np.minimum(
            np.maximum.accumulate(sorted_p * (m - np.arange(m))), 1.0
        )

        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = adjusted_sorted

        res_pvals_adjusted = adjusted
    elif adjust == "sidak":
        for i in res_pvals_unadjusted:
            res_pvals_adjusted.append(1 - (1 - i) ** m)
    elif adjust == "benjamini":
        # sort in ascending
        sorted_idx = np.argsort(res_pvals_unadjusted)
        # p_sorted = res_pvals_unadjusted[sorted_idx]
        p_sorted = np.sort(res_pvals_unadjusted)

        # compute m * p_sorted / rank  (rank = 1, 2, ..., m)
        ranks = np.arange(1, m + 1, dtype=float)
        raw = m * p_sorted / ranks

        # enforce monotonicity via cumulative min from the right
        # (this is the min_{j >= i} in the formula)
        adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]

        # cap at 1
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

        # map back to original order
        adjusted = np.empty(m)
        adjusted[sorted_idx] = adjusted_sorted

        res_pvals_adjusted = adjusted
    else:  # none
        res_pvals_adjusted = res_pvals_unadjusted.copy()

    # now return
    return DunnsResult(
        comparisons=res_groups, statistic=np.array(res_stats), pvalue=res_pvals_adjusted
    )


def brown(
    *samples: np.typing.ArrayLike,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> BrownResult:
    """
    Runs the Brown-Forsythe Robust ANOVA test. This is typically used when assumptions of One Way ANOVA are not met.

    This function is *not* to be confused with the *Brown-Forsythe Test for Homogeneity of Variances* which checks for equal variance.

    Parameters
    ----------
    *samples : array_like
        The sample measurements for each group. There must be at least two arguments.
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle input NaNs.

    Returns
    -------
    statistic : float
        The F*-Statistic
    pvalue : float
        The computed p-value

    References
    ----------
    [1] Brown, Morton B., and Alan B. Forsythe. "Robust Tests for the Equality of Variances." Journal of the American Statistical Association, vol. 69, no. 346, Informa UK Limited, June 1974, pp. 364-67, https://doi.org/10.1080/01621459.1974.10482955.
    """
    samples = [np.asarray(i) for i in samples]

    samples = np.array(samples)

    # handle nan input
    has_nan = np.isnan(samples).any()
    if has_nan:
        if nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        if nan_policy == "propagate":
            return BrownResult(statistic=np.nan, pvalue=np.nan)
        samples = samples[~np.isnan(samples)]

    # compute grand mean
    grand_mean = np.mean(samples)

    n, p = samples.shape

    # compute mean of each group
    group_means = np.mean(samples, axis=1)

    numerator = np.sum(p * ((group_means - grand_mean) ** 2))

    # compute group variances
    group_variances = np.var(samples, axis=1, ddof=1)

    # compute total items in dataset
    N = n * p

    denominator = np.sum((1 - p / N) * group_variances)

    f_star = numerator / denominator

    # compute degrees of freedom
    df_numerator = n - 1

    ci = 1 - p / N
    ci2 = ci**2
    s4 = group_variances**2
    df_denominator = (np.sum(ci * group_variances)) ** 2 / np.sum(ci2 * s4 / (p - 1))

    p_value = f.sf(f_star, df_numerator, df_denominator)

    return BrownResult(statistic=f_star, pvalue=p_value)


def _greenhouse_epsilon(samples: np.ndarray) -> float:
    """
    This is a helper function for `anovapy.oneway.f_repeated`. It really just computes the epsilon used for adjusting the degrees of freedom.

    Parameters
    ----------
    samples : np.ndarray
        All samples

    Returns
    -------
    The epsilon

    References
    ----------
    .. [1] Greenhouse, S.W., & Geisser, S. (1959). On methods in the analysis of profile data. Psychometrika, 24, 95-112.
    .. [2] Zaiontz, Charles. "Correcting for Sphericity." Real-Statistics.Com, 2015, https://real-statistics.com/anova-repeated-measures/sphericity/. Accessed 11 Nov. 2026.
    """
    # formula here is: eps = (sum(eigen_values_of_samples_covariance_matrix))^2 / (k-1)(sum(eigen_values_sample_covariance_matrix^2))
    # first we need to make a transformed covariance matrix of orthogonalized treatment contrasts
    x = np.column_stack(samples)

    S = np.cov(x, rowvar=False)
    M = helmert(x.shape[1])
    C = M @ S @ M.T

    # now we can apply it directly to the formula
    numerator = np.trace(C) ** 2
    denominator = (x.shape[1] - 1) * np.trace(C @ C)

    return min(1, numerator / denominator)  # cap at 1 just in case


def _huynh_epsilon(samples: np.ndarray) -> float:
    """
    Another helper function for `anovapy.oneway.f_repeated`.

    Parameters
    ----------
    samples : np.ndarray
        All samples

    Returns
    -------
    The epsilon

    References
    ----------
    .. [1] Huynh Huynh, & Feldt, L. S. (1976). Estimation of the Box Correction for Degrees of Freedom from Sample Data in Randomized Block and Split-Plot Designs. Journal of Educational Statistics, 1(1), 69-82. https://doi.org/10.2307/1164736
    .. [2] Zaiontz, Charles. "Correcting for Sphericity." Real-Statistics.Com, 2015, https://real-statistics.com/anova-repeated-measures/sphericity/. Accessed 11 Nov. 2026.
    """
    # epsilon formula here is: eps = (n(k-1)eps_gg - 2) / ((k-1)(n-1-(k-1)eps_gg))
    # where eps_gg is the computed epsilon from the Greenhouse-Geisser formula
    # n is the total number of subjects
    # and k is the total number of repeated measures
    # first compute the Greenhouse-Geisser epsilon
    eps_gg = _greenhouse_epsilon(samples)
    k, N = samples.shape
    # now apply to formula
    eps = (N * (k - 1) * eps_gg - 2) / ((k - 1) * (N - 1 - (k - 1) * eps_gg))

    return min(1, eps)  # cap at 1


def f_repeated(
    *samples: np.typing.ArrayLike,
    correction: Literal["greenhouse", "huynh", "lower"] | None = None,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> RepeatedMeasuresAnovaResult:
    """
    Performs a One Way Repeated Measures ANOVA.

    The One Way Repeated Measures ANOVA tests whether there are statistically significant differences between the means of three or more related groups where the same subjects are measured multiple times. It is used to detect changes in a dependent variable across different time points or under different conditions for the same participants, effectively extending the paired-samples t-test to more than two groups.

    Parameters
    ----------
    *samples : array_like
        Samples measurements for each group.
    correction: {'greenhouse', 'huynh', 'lower', None}
        Whether to apply any corrections. Note that this is usually done with the sphericity assumption is violated. When `None` here is the value, it means the sphericity assumption was met.
        - None : do no correction; the sphericity assumption was met
        - 'greenhouse' : Apply the Greenhouse-Geisser correction
        - 'huynh' : Apply the Huynh-Feldt correction
        - 'lower' : Apply the Lower-bound correction
    nan_policy : {'propagate', 'omit', 'raise'}
        How to handle NaN inputs. Defaults to 'propagate'

    Returns
    -------
    statistic : float
        The computed F statistic of the test
    pvalue : float
        The associated p-value from the F-distribution

    Notes
    -----
    This function assumes *sphericity* -- this can be checked with the Mauchly Test (see `anovapy.sphericity.mauchly`). When this assumption is not met, a correction may be applied via the `correction` parameter.

    This function is not to be confused with a One Way ANOVA (see `scipy.stats.f_oneway`). Whereas a One Way ANOVA compares means across **three or more independent groups** (different subjects in each group), the One Way Repeated Measures ANOVA compares means across **three or more related groups** (the same subjects measured repeatedly under different conditions or time points). For more info on this, check out: https://www.statology.org/one-way-anova-vs-repeated-measures-anova/

    References
    ----------
    .. [1] Keppel, G., & Wickens, T. D. (2004). Design and Analysis in Research (4th ed.). Pearson.
    .. [2] Zach. "How to Perform a Repeated Measures ANOVA by Hand." Statology, 8 May 2020, https://www.statology.org/repeated-measures-anova-by-hand/. Accessed 11 Nov. 2026.
    """
    samples = np.asarray(samples)

    # validate
    if correction not in {"greenhouse", "huynh", "lower", None}:
        raise ValueError(
            "correction must be one of {None, 'greenhouse', 'huynh', 'lower'}."
        )
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # handle nan
    has_nan = np.isnan(samples).any()
    if has_nan:
        if nan_policy == "propagate":
            return RepeatedMeasuresAnovaResult(statistic=np.nan, pvalue=np.nan)
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        # omit
        samples = samples[:, ~np.isnan(samples).any(axis=0)]

    n, p = samples.shape

    # first calculate total sum of squares
    # sst = s^2(n_total - 1)
    # where s^2 is variance and n_total refers to total number of values in the data
    s2 = np.var(samples, ddof=1)
    n_total = n * p
    sst = s2 * (n_total - 1)

    # calculate sum of squares between
    # ssb = sum(n_j * (mean_j_group - grand_mean) ** 2)
    # compute means for each group then apply the formula simply
    x_j = np.mean(samples, axis=1)
    grand_mean = np.mean(samples)
    ssb = np.sum(p * ((x_j - grand_mean) ** 2))

    # calculate subject sum of squares
    # sss = (sum(r^2_k / c)) - (N^2 / rc)
    # where r^2_k is the squared sum of the kth variable
    # c is n and r is p
    # and N is the total of the dataset
    N = np.sum(samples)
    # we only need to compute r^2_k
    r_sum = np.sum(samples, axis=0)
    sss = (np.sum(r_sum**2) / n) - ((N**2) / (n * p))

    # calculate sum of squares error
    # sse = sst - ssb - sss
    sse = sst - ssb - sss

    # compute degrees of freedom
    df_between = n - 1
    df_subject = p - 1
    df_error = df_between * df_subject
    ms_between = ssb / df_between
    # ms_subject = sss / df_subject
    ms_error = sse / df_error
    f_stat = ms_between / ms_error

    # check if we have any kind of correction to do
    if correction == "greenhouse":
        eps = _greenhouse_epsilon(samples)
        # we adjust the degrees of freedom
        df_between = df_between * eps
        df_error = df_error * eps
    elif correction == "huynh":
        eps = _huynh_epsilon(samples)
        df_between = df_between * eps
        df_error = df_error * eps
    elif correction == "lower":
        # formula here is 1 / (k - 1)
        # where k is the number of repeated measures
        eps = 1 / (n - 1)
        df_between = df_between * eps
        df_error = df_error * eps

    p_value = f.sf(f_stat, df_between, df_error)

    return RepeatedMeasuresAnovaResult(statistic=f_stat, pvalue=p_value)


def pairwise_ttest(
    *samples: np.typing.ArrayLike,
    adjust: Literal["bonferroni", "holm", "none"] = "bonferroni",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> AdjustedTTestResult:
    """
    Runs a pairwise paired t-test with p-value correction. This is typically used as a post-hoc test after running a One Way Repeated Measures ANOVA (see `anovapy.oneway.f_repeated`).

    Parameters
    ----------
    *samples : array_like
        Sample measurments for each groups.
    adjust : {'bonferroni', 'holm', 'none'}
        How p-values should be adjusted. Defaults to 'bonferroni'
        - 'bonferroni' : Adjusts p-values with the Bonferroni method. This is most commonly used (hence calling the test a Bonferroni-adjusted paired t-test)
        - 'holm' : Adjusts p-values with the Holm-Bonferroni method (i.e. Holm corrected paired t-test)
        - 'none' : Do not adjust the p-value
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle inputs with NaN values.

    Returns
    -------
    AdjustedTTestResult
        A dataclass containing all computed t-statistics for each comparison at index i, j and the computed p-values.

    References
    ----------
    .. [1] Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics, 6, 65-70.
    """
    samples = np.asarray(samples)

    # validate
    if adjust not in {"bonferroni", "holm", "none"}:
        raise ValueError("adjust must be one of {'bonferroni', 'holm', 'none'}.")
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # handle nan
    has_nan = np.isnan(samples).any()
    if has_nan:
        if nan_policy == "propagate":
            return AdjustedTTestResult(
                statistic=np.array(np.nan), pvalue=np.array(np.nan)
            )
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        # omit
        samples = samples[:, ~np.isnan(samples).any(axis=0)]

    # this is pretty easy
    # for each pair just run SciPy's related t-test on them
    # then adjust those p-values
    stats = []
    pvals = []

    for i in range(len(samples)):
        for z in range(i + 1, len(samples)):
            # run t-test
            res = ttest_rel(samples[i], samples[z])
            stats.append(res.statistic)
            pvals.append(res.pvalue)

    # now adjust
    m = len(pvals)
    pvals = np.array(pvals)
    adjusted = []
    if adjust == "bonferroni":
        for k in pvals:
            adjusted.append(min(1.0, k * m))
    elif adjust == "holm":
        order = np.argsort(pvals)
        sorted_p = pvals[order]

        adjusted_sorted = np.minimum(
            np.maximum.accumulate(sorted_p * (m - np.arange(m))), 1.0
        )

        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = adjusted_sorted
    else:  # none
        adjusted = pvals

    return AdjustedTTestResult(statistic=np.array(stats), pvalue=np.array(adjusted))
