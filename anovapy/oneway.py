# everything here is purly scipy just with its own functions
# and naming conventions for simplicity
# a few more functions though are added

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import helmert
from scipy.stats import f, f_oneway, norm, rankdata, ttest_rel

__all__ = [
    "brown",
    "dunns",
    "f_ancova",
    "f_permutation",
    "f_repeated",
    "pairwise_ttest",
]

BrownResult = namedtuple("BrownResult", ["statistic", "pvalue"])
RepeatedMeasuresAnovaResult = namedtuple(
    "RepeatedMeasuresAnovaResult", ["statistic", "pvalue"]
)
F_permutationResult = namedtuple("F_permutationResult", ["statistic", "pvalue"])


@dataclass
class F_ancovaResult:
    """
    Result of `anovapy.oneway.f_ancova`.

    Attributes
    ----------
    statistic : float
        F-statistic for the independent factor
    pvalue : float
        Computed p-value for the independent factor
    df_factor : int
        Degrees of Freedom for Factor
    df_error : int
        Degrees of Freedom Error
    covariate_stats : np.ndarray
        1D array of F-statistics for each covariate passed in
    covariate_pvalues : np.ndarray
        1D array of p-values for each covariate passed in
    """

    statistic: float
    pvalue: float
    df_factor: int
    df_error: int
    covariate_stats: np.ndarray
    covariate_pvalues: np.ndarray


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


def f_ancova(
    independent: np.typing.ArrayLike,
    dependent: np.typing.ArrayLike,
    *covariates: np.typing.ArrayLike,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> F_ancovaResult:
    """
    Performs a One Way Analysis of Covariance (ANCOVA).

    Parameters
    ----------
    independent : array_like
        A 1D array representing a categorical variable. This *must* have at least two or more levels.
    dependent : array_like
        A 1D array representing a continuous dependent variable.
    *covariates : array_like
        At least one covariate. Each must be a 1D array
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle input NaNs.

    Returns
    -------
    F_ancovaResult
        A dataclass containing the F-statistic and p-value for the independent factor, along with degrees of freedom and computed F-statistic and p-value for each covariate passed in.

    Notes
    -----
    The general formula here is:

    $Y_{ij} = u + T_i + B(X_{ij} - X) + e_{ij}$

    where *Y_ij* is the dependent variable for the j-th individual in the i-th group, *u* is the grand mean, *T_i* is the fixed treatment effect for group i, *B* is the regression coefficient (slope) representing the linear relationship between the covariate and the dependent variable, *X_ij* is the value of the continuous covariate for the j-th individual in group i, *X* is the overall mean of the covariate, and *e_ij* is random error term, assumed to be normally distributed with mean zero.

    Sometimes the formula above is shown in matrix multiplication, which can be represented as:

    $Y = B_0 + B_1X + B_2C + e$

    where *B_0* is the predicted Y for the reference group when all covariates = 0, *B_1* is first group dummy coefficient (or first covariate slope, depending on your column ordering), *X* is the design matrix shown above (not a single covariate — it's the entire matrix of predictors), and *B_2* is the covariate slope.

    At the moment, this function strictly implements **Type II Sum of Squares.**

    References
    ----------
    .. [1] Fisher, R. A. (1932). Statistical Methods for Research Workers (4th ed.). Edinburgh: Oliver & Boyd.
    .. [2] Khammar A, Yarahmadi M, Madadizadeh F. What Is Analysis of Covariance (ANCOVA) and How to Correctly Report Its Results in Medical Research? Iran J Public Health. 2020 May;49(5):1016-1017. PMID: 32953697; PMCID: PMC7475615.
    .. [3] Wikipedia Contributors. "Analysis of Covariance." Wikipedia, Wikimedia Foundation, 9 June 2019, https://en.wikipedia.org/wiki/Analysis_of_covariance. Accessed 12 Nov. 2026.
    .. [4] Bobbitt, Zach. "An Introduction to ANCOVA (Analysis of Variance)." Statology, 19 May 2020, https://www.statology.org/ancova/. Accessed 13 Nov. 2026.
    .. [5] Keller, Bryan, and Dobrin Marchev. Running Head: ANCOVA 1 Analysis of Covariance: Univariate and Multivariate Approaches Forthcoming in the International Encyclopedia of Education, 4Th Edition. 2022, http://www.columbia.edu/~bsk2131/Keller_Marchev_ANCOVA_R1.pdf. Accessed 12 Nov. 2026.
    .. [6] NCSS, LLC. One-Way Analysis of Covariance (ANCOVA). Technical procedure guide, NCSS, LLC, 4 Oct. 2024, https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/NCSS/One-Way_Analysis_of_Covariance-ANCOVA.pdf. Accessed 12 Nov. 2026.
    """
    independent = np.asarray(independent)
    dependent = np.asarray(dependent, dtype=float)
    covariates = [np.asarray(c, dtype=float) for c in covariates]

    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # handle nan
    # TODO: CHECK NAN IN COVARIATES AND OMIT RESULTS
    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "propagate":
            return F_ancovaResult(
                statistic=np.nan,
                pvalue=np.nan,
                df_factor=np.nan,
                df_error=np.nan,
                covariate_stats=np.array(np.nan),
                covariate_pvalues=np.array(np.nan),
            )
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        # omit

    # validate that independent variable has at least two levels
    unique_groups = np.unique(independent)

    if len(unique_groups) == 1:
        raise ValueError("The independent variable must have at least two levels.")

    # lets utilize the formula that involves matrix multiplication as it is easier to work with

    n = len(dependent)
    g = len(unique_groups)

    # construct Design Matrix via General Linear Model (GLM) approach
    # drop teh first dummy variables for the categorical independent factor groups
    dummy_cols = [(independent == grp).astype(float) for grp in unique_groups[1:]]
    D = np.column_stack(dummy_cols) if dummy_cols else np.empty((n, 0))

    # assemble full design matrix: [Intercept (B_0) | Group Dummies (T_i) | Covariates (C)]
    intercept = np.ones((n, 1))
    C = np.column_stack(covariates) if covariates else np.empty((n, 0))
    # now combine column, group dummies, and continuous covariates
    X_full = np.hstack([intercept, D, C])

    # fit full model via Ordinary Least Squares (OLS)
    # solves min ||X_full * beta - dependent||^2 to find Residual Sum of Squares (RSS_full)
    beta_full, rss_full, _, _ = np.linalg.lstsq(X_full, dependent, rcond=None)
    # normalize rss_full format across different NumPy versions (scalar vs 1D array)
    if rss_full.ndim > 0 and len(rss_full) > 0:
        rss_full = rss_full[0]
    elif rss_full.ndim == 0:
        pass  # rss_full is already a scalar!
    else:
        rss_full = np.sum((dependent - (X_full @ beta_full)) ** 2)

    # compute degrees of freedom error
    df_error = n - X_full.shape[1]
    if df_error <= 0:
        raise ValueError(
            "Degrees of freedom for error must be greater than 0. Too many parameters for sample size."
        )
    # now compute mean squared error
    ms_error = rss_full / df_error

    # now compute sum of squares
    # sstotal = np.sum((dependent - np.mean(dependent)) ** 2)
    # df_total = n - 1

    # F-Test for Independent Factor (Group Effect)
    # reduced model drops only the group dummy columns: [Intercept, Covariates]
    X_red_factor = np.hstack([intercept, C])
    _, rss_red_factor, _, _ = np.linalg.lstsq(X_red_factor, dependent, rcond=None)
    if rss_red_factor.ndim > 0 and len(rss_red_factor) > 0:
        rss_red_factor = rss_red_factor[0]

    # Sum of Squares for the Factor is the increase in RSS when groups are removed
    ss_factor = max(0.0, rss_red_factor - rss_full)
    df_factor = g - 1
    ms_factor = ss_factor / df_factor if df_factor > 0 else 0.0
    # now we can compute the actual F-statistic for the factor and p-value
    # just use the same formula as always
    f_factor = ms_factor / ms_error if ms_error > 0 else 0.0
    p_factor = f.sf(f_factor, df_factor, df_error)

    # now let's compute the F-statistic and p-value for each covariate
    covariate_stats = np.zeros(len(covariates))  # to be filled
    covariate_pvals = np.zeros(len(covariates))  # to be filled
    for i in range(len(covariates)):
        # isolate and drop the j-th covariate column from the full design matrix
        col_idx = 1 + D.shape[1] + i
        X_red_cov = np.delete(X_full, col_idx, axis=1)

        # fit reduced model without ith covariate
        _, rss_red_cov, _, _ = np.linalg.lstsq(X_red_cov, dependent, rcond=None)
        if rss_red_cov.ndim > 0 and len(rss_red_cov) > 0:
            rss_red_cov = rss_red_cov[0]

        # covariate Sum of Squares is the increase in RSS resulting from dropping ith covariate
        ss_cov = max(0.0, rss_red_cov - rss_full)
        df_cov = 1
        ms_cov = ss_cov / df_cov
        f_cov = ms_cov / ms_error if ms_error > 0 else 0.0
        p_cov = f.sf(f_cov, df_cov, df_error)

        covariate_stats[i] = f_cov
        covariate_pvals[i] = p_cov

    return F_ancovaResult(
        statistic=f_factor,
        pvalue=p_factor,
        df_factor=df_factor,
        df_error=df_error,
        covariate_stats=covariate_stats,
        covariate_pvalues=covariate_pvals,
    )


def f_permutation(
    *samples: np.typing.ArrayLike,
    permutations: int = 999,
    random_state: int | np.random.Generator | None = None,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> F_permutationResult:
    """
    Runs a One Way Permutation ANOVA.

    This can be used when assumptions for a One Way ANOVA are not met.

    Parameters
    ----------
    *samples : array_like
        The sample measurements for each group. There must be at least two arguments.
    permutations : int
        Number of permutations to run. Defaults to 999
    random_state : int | np.random.Generator | None
        A seed to set for reproducible results
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle input NaNs.

    Returns
    -------
    statistic : float
        Computed F-statistic
    pvalue : float
        Computed p-value

    Notes
    -----
    A note on comparing this function to an R library like `wPerm`. AnovaPy uses NumPy to shuffle and generate random data. NumPy uses a different algorithm compared to R when generating random data, and hence, when comparing this function to R, p-values may look slightly different. Hence, do not expect p-values to match R's wPerm package seed-for-seed.
    During testing of this function, statistical stability and convergence were tested, not identical pseudo-random streams.

    The `permutations` parameter controls the number of random label shuffles used to build the null distribution of the F-statistic. Leaving it at its default, `999`, is recommended as it balances runtime speed with enough resolution to clear standard significance thresholds.

    References
    ----------
    .. [1] Anderson, Marti J. "Permutation Tests for Univariate or Multivariate Analysis of Variance and Regression." Canadian Journal of Fisheries and Aquatic Sciences, vol. 58, no. 3, Mar. 2001, pp. 626-39, https://doi.org/10.1139/f01-004.
    """
    samples = [np.asarray(i) for i in samples]

    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy not in {'propagate', 'omit', 'raise'}.")

    # handle nan
    # TODO: MAYBE IMPROVE THIS SOMEHOW?
    for i in range(len(samples)):
        has_nan = np.isnan(samples[i]).any()
        if has_nan:
            if nan_policy == "propagate":
                return F_permutationResult(statistic=np.nan, pvalue=np.nan)
            elif nan_policy == "raise":
                raise ValueError("Input contains NaN values.")
            samples[i] = samples[i][~np.isnan(samples[i])]

    # first compute observed statistic
    F_obs, _ = f_oneway(*samples)

    # pool all data and record group sizes
    pooled = np.concatenate(samples)
    sizes = [len(s) for s in samples]

    # now shuffle the data n times (as given by the function) and compute the F-statistic for each
    F_perm = np.zeros(permutations)

    gen = np.random.default_rng(random_state)
    for i in range(permutations):
        # shuffle pooled data
        shuffled = gen.permutation(pooled)
        # resplit into groups of original sizes
        groups = []
        start = 0
        for n in sizes:
            groups.append(shuffled[start : start + n])
            start += n
        # compute f-stat on that
        F_p, _ = f_oneway(*groups)
        F_perm[i] = F_p

    # now compute p-value
    p_value = (np.sum(F_perm >= F_obs) + 1) / (permutations + 1)

    return F_permutationResult(statistic=F_obs, pvalue=p_value)
