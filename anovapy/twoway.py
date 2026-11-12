# two-way anova along with post hoc tests
# and alternative tests

import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Literal

import numpy as np
from scipy.stats import boxcox, chi2, f, rankdata, studentized_range, t, ttest_ind

__all__ = [
    "art",
    "artc",
    "f_twoway",
    "pairwise_welch",
    "scheirer",
    "transform_and_run_twoway",
    "transform_dependent",
    "tukey_hsd",
]


@dataclass(frozen=True)
class ScheirerResult:
    """
    Result of `anovapy.twoway.scheirer`.

    Attributes
    ----------
    statistic_a : float
        H-Statistic of Factor A
    pvalue_a : float
        p-value of Factor A
    df_a : int
        Degrees of freedom for Factor A
    statistic_b : float
        H-Statistic of Factor B
    pvalue_b : float
        p-value of Factor B
    df_b : int
        Degrees of freedom for Factor B
    statistic_interaction : float
        H-Statistic of interaction
    pvalue_interaction : float
        p-value of interaction
    df_interaction : int
        Degrees of freedom for interaction
    """

    # H-statistic of Factor A
    statistic_a: float
    # p-value of Factor A
    pvalue_a: float
    # degrees of freedom for Factor A
    df_a: int
    # H-Statistic of Factor B
    statistic_b: float
    # p-value of Factor-B
    pvalue_b: float
    # degrees of freedom for Factor-B
    df_b: int
    # H-Statistic for Interaction
    statistic_interaction: float
    # p-value for interaction
    pvalue_interaction: float
    # degrees of freedom for interaction
    df_interaction: int

    def __repr__(self):
        return (
            f"ScheirerResult( "
            f"statistic_a={self.statistic_a}, "
            f"pvalue_a={self.pvalue_a}, "
            f"df_a={self.df_a}, "
            f"statistic_b={self.statistic_b}, "
            f"pvalue_b={self.pvalue_b}, "
            f"df_b={self.df_b}, "
            f"statistic_interaction={self.statistic_interaction}, "
            f"pvalue_interaction={self.pvalue_interaction}, "
            f"df_interaction={self.df_interaction}"
            ")"
        )


@dataclass(frozen=True)
class ARTResult:
    """
    Result of `anovapy.twoway.art`.

    Attributes
    ----------
    statistic_a : float
        The computed F-statistic of Factor A
    pvalue_a : float
        Computed p-value of Factor A
    df_num_a : int
        Numerator Degrees of Freedom of Factor A
    df_den_a : int
        Denominator (residual) Degrees of Freedom of Factor A
    statistic_b : float
        The computed F-statistic of Factor B
    pvalue_b : float
        Computed p-value of Factor B
    df_num_b : int
        Numerator Degrees of Freedom of Factor B
    df_den_b : int
        Denominator (residual) Degrees of Freedom of Factor B
    statistic_interaction : float
        The computed F-statistic of Interaction
    pvalue_interaction : float
        Computed p-value of Interaction
    df_num_interaction : int
        Numerator Degrees of Freedom of Interaction
    df_den_interaction : int
        Denominator (residual) Degrees of Freedom of Interaction
    """

    statistic_a: float
    pvalue_a: float
    df_num_a: int
    df_den_a: int

    statistic_b: float
    pvalue_b: float
    df_num_b: int
    df_den_b: int

    statistic_interaction: float
    pvalue_interaction: float
    df_num_interaction: int
    df_den_interaction: int


@dataclass(frozen=True)
class ARTContrastResult:
    """
    Result for `anovapy.twoway.artc`.

    Attributes
    ----------
    comparison : list[tuple]
        A list of all comparisons made. Each element contains a tuple with two values highlighting the comparison being made.
    statistic : list[float]
        Computed t-statistic for each comparison
    pvalue : list[float]
        Computed (and adjusted) p-value for each comparison
    df : list[int]
        Degrees of Freedom for each comparison
    """

    comparisons: list[tuple[Any, Any]]
    statistic: list[float]
    pvalue: list[float]
    df: list[int]


@dataclass
class TwoWayAnovaResult:
    """
    Data class storing two-way ANOVA results for Factor A, Factor B,
    and their Interaction.

    Attributes
    ----------
    statistic_a : float
        The F-statistic for Factor A
    pvalue_a : float
        Computed p-value of Factor A
    df_a : int
        Degrees of Freedom for Factor A
    statistic_b : float
        The F-Statistic for Factor B
    pvalue_b : float
        Computed p-value of Factor B
    df_b : int
        Degrees of Freedom for Factor B
    statistic_interaction : float
        The F-Statistic for the Interaction
    pvalue_interaction : float
        Computed p-value of Interaction
    df_interaction : int
        Degrees of Freedom for Interaction
    """

    # Factor A results
    statistic_a: float
    pvalue_a: float
    df_a: int

    # Factor B results
    statistic_b: float
    pvalue_b: float
    df_b: int

    # Interaction results
    statistic_interaction: float
    pvalue_interaction: float
    df_interaction: int

    def __repr__(self):
        return (
            f"TwoWayAnovaResult("
            f"  statistic_a={self.statistic_a}, "
            f"  pvalue_a={self.pvalue_a}, "
            f"  df_a={self.df_a}, "
            f"  statistic_b={self.statistic_b}, "
            f"  pvalue_b={self.pvalue_b}, "
            f"  df_b={self.df_b}, "
            f"  statistic_interaction={self.statistic_interaction}, "
            f"  pvalue_interaction={self.pvalue_interaction}, "
            f"  df_interaction={self.df_interaction}"
            f")"
        )


@dataclass
class TukeyHSDResult:
    """
    Result of the Tukey HSD Post-Hoc Test (see `anovapy.twoway.tukey_hsd`).

    Each attribute contains an array -- all of the same length -- where index `i` represents a comparison.

    Attributes
    ----------
    comparisons : np.ndarray
        A 1D array containing all comparisons. Each element contains a tuple with two values highlighting the comparison being made.
    statistic : np.ndarray
        A 1D array containing all Q-statistic values
    pvalue : np.ndarray
        A 1D array containing all p-values
    """

    comparisons: np.ndarray
    statistic: np.ndarray
    pvalue: np.ndarray

    def __repr__(self):
        return (
            f"TukeyHSDResult( "
            f"  comparisons={self.comparisons}, "
            f"  statistic={self.statistic}, "
            f"  pvalue={self.pvalue}"
            f")"
        )


@dataclass
class WelchResult:
    """
    Result of `anovapy.twoway.welch`.

    Attributes
    ----------
    comparisons : np.ndarray
        All comparisons made during the test
    statistic : np.ndarray
        All computed t-statistics
    pvalue : np.ndarray
        Adjusted p-value via the Holm-Bonferroni method
    """

    comparisons: np.ndarray
    statistic: np.ndarray
    pvalue: np.ndarray


def f_twoway(
    factor_a: np.typing.ArrayLike,
    factor_b: np.typing.ArrayLike,
    dependent: np.typing.ArrayLike,
    *,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> TwoWayAnovaResult:
    """
    Performs a Two-Way ANOVA.

    Note that this test assumes normality and homogeneity of variance.
    Sums of squares are computed in the order Factor A, Factor B, then their
    interaction. For unbalanced designs, changing the factor order can change
    the main-effect sums of squares.

    Parameters
    ----------
    factor_a : array_like
        The first factor.
    factor_b : array_like
        The second factor.
    dependent : array_like
        The dependent variable.
    nan_policy : {'propagate', 'omit', 'raise'}
        How to handle NaN inputs. Defaults to 'propagate'

    Returns
    -------
    TwoWayAnovaResult
        Contains the computed statistic and p-values

    Raises
    ------
    ValueError if `nan_policy` is invalid or if there is a cell that has only one observation.

    Notes
    -----
    This function implements the Type I Sum of Squares when computing the F-Statistic.

    References
    ----------
    .. [1] Kim HY. Statistical notes for clinical researchers: Two-way analysis of variance (ANOVA)-exploring possible interaction between factors. Restor Dent Endod. 2014 May;39(2):143-7. doi: 10.5395/rde.2014.39.2.143. PMID: 24790929; PMCID: PMC3978106.
    .. [2] Bobbitt, Zach. "How to Perform a Two-Way ANOVA by Hand." Statology, 30 Nov. 2021, https://www.statology.org/two-way-anova-by-hand/. Accessed 11 Nov. 2026.
    """
    factor_a = np.asarray(factor_a, dtype=str)
    factor_b = np.asarray(factor_b, dtype=str)
    dependent = np.asarray(dependent)

    # validate nan policy
    has_nan = np.isnan(dependent).any()
    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")
    if has_nan:
        if nan_policy == "propagate":
            return TwoWayAnovaResult(
                statistic_a=np.nan,
                pvalue_a=np.nan,
                df_a=np.nan,
                statistic_b=np.nan,
                pvalue_b=np.nan,
                df_b=np.nan,
                statistic_interaction=np.nan,
                pvalue_interaction=np.nan,
                df_interaction=np.nan,
            )
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        # omit
        dependent = dependent[~np.isnan(dependent)]

    # first compute grand mean
    grand_mean = np.mean(dependent)

    # compute sum of squares for factor_a
    # compute mean of each part in factor to help with this
    unique_vals_a = np.unique(factor_a)
    x_j = []
    subject_indecies_a: dict[str, list[int]] = {}
    n_a = []  # to be set, refers to sample size at group j in factor a
    for i in unique_vals_a:
        indecies = np.where(factor_a == i)[0]
        subject_indecies_a[i] = indecies
        items = []  # list of all float values associated with it
        for z in indecies:
            items.append(dependent[z])
        n_a.append(len(items))
        items = np.array(items)
        x_j.append(np.mean(items))

    x_j = np.array(x_j)

    n_a = np.array(n_a)

    # sum of squares = n * sum(X_j - grand_mean) ** 2
    # where X_j are computed means
    # and n is the sample size of group j
    ss_a = np.sum(n_a * ((x_j - grand_mean) ** 2))

    # do the same for the second factor
    unique_vals_b = np.unique(factor_b)
    x_j = []
    subject_indecies_b: dict[str, list[int]] = {}
    n_b = []  # refers to sample size of group j in factor b
    for i in unique_vals_b:
        indecies = np.where(factor_b == i)[0]
        subject_indecies_b[i] = indecies
        items = []
        for z in indecies:
            items.append(dependent[z])
        n_b.append(len(items))
        items = np.array(items)
        x_j.append(np.mean(items))

    x_j = np.array(x_j)

    n_b = np.array(n_b)

    ss_b = np.sum(n_b * ((x_j - grand_mean) ** 2))

    # calculate sum of squares within (error)
    # this one works by making a combination of both factors in short
    ss_within = 0  # to be computed
    for i in unique_vals_a:
        for z in unique_vals_b:
            # get values explicitly within this range
            vals_i = subject_indecies_a[i]
            vals_z = subject_indecies_b[z]
            # we have to compute sum of squares here first
            # so get all values
            common_values = np.intersect1d(vals_i, vals_z)
            vals = [dependent[c] for c in common_values]
            # validate cell size as this is critical
            if len(vals) == 1:
                raise ValueError(
                    f'Two Way ANOVA requires at least two observations per cell. Cell at "{i}", "{z}" contains only one observation.'
                )
            vals = np.array(vals)
            valsm = np.mean(vals)
            ss_within += np.sum((vals - valsm) ** 2)

    # calculate total sum of squares
    ss_total = np.sum((dependent - grand_mean) ** 2)

    # calculate sum of squares interaction
    # SS Interaction = SS Total – SS Factor 1 – SS Factor 2 – SS Within
    ss_interaction = ss_total - ss_a - ss_b - ss_within

    # compute degrees of freedom for each
    df_a = len(unique_vals_a) - 1
    df_b = len(unique_vals_b) - 1
    df_interaction = df_a * df_b
    df_within = len(dependent) - (len(unique_vals_a) * len(unique_vals_b))

    # compute ms within for each
    ms_a = ss_a / df_a
    ms_b = ss_b / df_b
    ms_interaction = ss_interaction / df_interaction
    ms_within = ss_within / df_within

    # now compute f-stat
    # f = ms_[factor] / ms_within
    f_a = ms_a / ms_within
    f_b = ms_b / ms_within
    f_interaction = ms_interaction / ms_within

    # now compute p-value
    p_a = f.sf(f_a, df_a, df_within)
    p_b = f.sf(f_b, df_b, df_within)
    p_interaction = f.sf(f_interaction, df_interaction, df_within)

    return TwoWayAnovaResult(
        statistic_a=f_a,
        pvalue_a=p_a,
        df_a=df_a,
        statistic_b=f_b,
        pvalue_b=p_b,
        df_b=df_b,
        statistic_interaction=f_interaction,
        pvalue_interaction=p_interaction,
        df_interaction=df_interaction,
    )


def tukey_hsd(
    factor_a: np.typing.ArrayLike,
    factor_b: np.typing.ArrayLike,
    dependent: np.typing.ArrayLike,
    *,
    term: Literal["factor_a", "factor_b", "interaction"] = "factor_a",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> TukeyHSDResult:
    """
    Performs Tukey's HSD Post-Hoc test.

    This is usually done after results are found statistically significant in a Two Way ANOVA.

    Parameters
    ----------
    factor_a : array_like
        1D array containing categorical labels for Factor A.
    factor_b : array_like
        1D array containing categorical labels for Factor B.
    dependent : array_like
        1D array containing continuous dependent observations.
    term : {'factor_a', 'factor_b', 'interaction'}, default: 'factor_a'
        Determines which main effect or interaction to analyze:
        - 'factor_a': Pairwise comparisons across Factor A levels.
        - 'factor_b': Pairwise comparisons across Factor B levels.
        - 'interaction': Pairwise comparisons across all Factor A x B cell combinations.
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle NaN inputs. Defaults to 'propagate'

    Returns
    -------
    A dataclass, TukeyHSDResult, containing:
        - comparisons : All comparisons made
        - statistic : Q-statistics computed for each comparison (index i here is relevant to index i in `comparison`)
        - pvalue : All computed p-values
        - confidence_interval : The confidence interval

    Notes
    -----
    This post-hoc test is typically run after a result from Two Way ANOVA shows how it is statistically significant. Like with Two Way ANOVA, it also assumes normality and homogeneity of variance.

    This function has been implemented to be able to handle unequal cell sizes. When comparing unequal sample sizes, this function will default to using the Tukey-Kramer adjustment.

    Note that this function is not to be confused with `scipy.stats.tukey_hsd` which has been implemented to be run after a One Way ANOVA.

    References
    ----------
    .. [1] Tukey, John W. "Comparing Individual Means in the Analysis of Variance." Biometrics, vol. 5, no. 2, JSTOR, June 1949, p. 99, https://doi.org/10.2307/3001913. Accessed 11 Nov. 2026.
    .. [2] Libretexts. "13.7: Tests Supplementing." Statistics LibreTexts, Libretexts, 2 Feb. 2022, https://stats.libretexts.org/Courses/Luther_College/Psyc_350%3A_Behavioral_Statistics_(Toussaint)/13%3A_Analysis_of_Variance/13.07%3A_Tests_Supplementing. Accessed 11 Nov. 2026.
    .. [3] Statistics Fundamentals. "Two-Way ANOVA Examples: Step-by-Step Calculations and Interpretation." Statisticsfundamentals.Com, Statistics Fundamentals, 14 Sept. 2026, https://statisticsfundamentals.com/hypothesis-testing/two-way-anova-examples/. Accessed 11 Nov. 2026.
    .. [4] Statistics Fundamentals. "Tukey HSD Test: Post-Hoc Comparisons After ANOVA." Statisticsfundamentals.Com, Statistics Fundamentals, 17 July 2026, https://statisticsfundamentals.com/hypothesis-testing/tukey-hsd-test/. Accessed 11 Nov. 2026.
    .. [5] Bobbitt, Zach. "What Is a Marginal Mean? (Definition & Example)." Statology, 12 Mar. 2021, https://www.statology.org/marginal-mean/. Accessed 11 Nov. 2026.

    Examples
    --------
    >>> from anovapy.twoway import tukey_hsd
    >>> import numpy as np
    >>> # just generate some random data for now
    >>> factor_a = np.repeat(['F1', 'F2', 'F3'], 4)
    >>> factor_b = np.tile(['Wheat', 'Rice'], 6)
    >>> dependent = np.random.normal(size=12)
    >>> # checking factor_a results
    >>> res_a = tukey_hsd(factor_a=factor_a, factor_b=factor_b, dependent=dependent, term='factor_a')
    >>> for i in range(len(res_a.statistic)):
    >>> ...     print(f"Comparison: {res_a.comparisons[i]}, Q-stat: {res_a.statistic[i]}, p-value: {res_a.pvalue[i]}")
    >>> # checking factor_b results
    >>> res_b = tukey_hsd(factor_a=factor_a, factor_b=factor_b, dependent=dependent, term='factor_b')
    >>> # checking interaction results
    >>> res_i = tukey_hsd(factor_a=factor_a, factor_b=factor_b, dependent=dependent, term='interaction')
    """
    factor_a = np.asarray(factor_a, dtype=str)
    factor_b = np.asarray(factor_b, dtype=str)
    dependent = np.asarray(dependent, dtype=float)

    # validate
    if term not in {"factor_a", "factor_b", "interaction"}:
        raise ValueError("term must be one of {'factor_a', 'factor_b', 'interaction'}.")
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "propagate":
            return TukeyHSDResult(comparisons=np.nan, statistic=np.nan, pvalue=np.nan)
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        dependent = dependent[~np.isnan(dependent)]

    # we first need to compute the mean squared error (MSE)
    # we can reuse the exact same formula as in f_twoway where we just need sum of squares within / degrees of freedom within
    unique_vals_a = np.unique(factor_a)
    unique_vals_b = np.unique(factor_b)

    # set up indecies
    subject_indecies_a: dict[str, list[int]] = {}
    subject_indecies_b: dict[str, list[int]] = {}
    for i in unique_vals_a:
        subject_indecies_a[i] = np.where(factor_a == i)[0]
    for i in unique_vals_b:
        subject_indecies_b[i] = np.where(factor_b == i)[0]

    ss_within = 0  # to be computed
    for i in unique_vals_a:
        for z in unique_vals_b:
            # get values explicitly within this range
            vals_i = subject_indecies_a[i]
            vals_z = subject_indecies_b[z]
            # we have to compute sum of squares here first
            # so get all values
            common_values = np.intersect1d(vals_i, vals_z)
            vals = [dependent[c] for c in common_values]
            # validate cell size as this is critical
            if len(vals) == 1:
                raise ValueError(
                    f'Tukey\'s HSD Two Way requires at least two observations per cell. Cell at "{i}", "{z}" contains only one observation.'
                )
            vals = np.array(vals)
            valsm = np.mean(vals)
            ss_within += np.sum((vals - valsm) ** 2)

    df_within = len(dependent) - (len(unique_vals_a) * len(unique_vals_b))
    ms_error = ss_within / df_within

    df_a = len(unique_vals_a)
    df_b = len(unique_vals_b)
    df_int = df_a * df_b

    res_comp: list[str] = []
    res_stats: list[float] = []
    res_pvals: list[float] = []

    # this next step regards which terms was selected
    if term == "factor_a" or term == "factor_b":
        # pick the right term to go over
        target = []  # to be filled
        subject_target: dict[str, list[int]] = {}  # to be filled
        df_target = 0  # to be filled

        if term == "factor_a":
            target = unique_vals_a
            subject_target = subject_indecies_a
            df_target = df_a
        else:
            target = unique_vals_b
            subject_target = subject_indecies_b
            df_target = df_b

        # go through each possible comparison
        for i in range(len(target)):
            for z in range(i + 1, len(target)):
                # comp: str = f"{target[i]}-{target[z]}"
                # res_comp.append(comp)
                res_comp.append((target[i], target[z]))
                # Q = m_i - m_j / sqrt(mse / n)
                # m_i and m_j are the marginal means of unique_vals_a[i] and unique_vals_a[z] respectively
                # mse is the mean squared error computed earlier
                # n refers to the size of all indecies
                indecies_i = subject_target[target[i]]
                indecies_z = subject_target[target[z]]
                n = len(indecies_i)
                denom = np.sqrt(ms_error / n)
                if len(indecies_i) != len(indecies_z):
                    # unequal length!
                    # apply Tukey-Kramer adjustment
                    # replace n with the harmonic mean of n_i and n_j
                    denom = np.sqrt(
                        ms_error / 2 * (1 / len(indecies_i) + 1 / len(indecies_z))
                    )

                # get marginal means
                items_i = []
                items_z = []

                for j in range(n):
                    items_i.append(dependent[indecies_i[j]])
                    items_z.append(dependent[indecies_z[j]])

                items_i = np.array(items_i)
                items_z = np.array(items_z)

                mi = np.mean(items_i)
                mj = np.mean(items_z)
                Q = abs((mi - mj) / denom)

                p = studentized_range.sf(Q, df_target, df_within)
                res_stats.append(Q)
                res_pvals.append(p)
    else:
        # interaction
        # mostly combining a whole bunch of stuff together
        # to make this go as smoothly as possible just map out all possible terms first
        def get_items_in_cell(label1: str, label2: str) -> np.ndarray:
            # helper for getting all items in a cell
            # label1 is assumed to be in factor_a indecies
            # label2 is assumed to be in factor_b indecies
            vals1 = subject_indecies_a[label1]
            vals2 = subject_indecies_b[label2]
            # intersect them
            union = np.intersect1d(np.array(vals1), np.array(vals2))
            # now collect items
            items = []
            for i in union:
                items.append(dependent[i])
            return np.array(items)

        # get all combinations
        com = list(product(unique_vals_a, unique_vals_b))
        # go through each combination possible
        for i in range(len(com)):
            for z in range(i + 1, len(com)):
                labeli_1, labeli_2 = com[i]
                labelz_1, labelz_2 = com[z]
                # comp: str = f"{labeli_1}:{labeli_2}-{labelz_1}:{labelz_2}"
                # res_comp.append(comp)
                res_comp.append(((labeli_1, labeli_2), (labelz_1, labelz_2)))

                # compute q-stat and p-value for comp1 first
                # the formula here isn't that different
                # only thing that changes is:
                # Q = (Y_ik - Y_jk) / sqrt(MSE / r)
                # where r refers to the number of items in a cell
                y_ik = get_items_in_cell(labeli_1, labeli_2)
                y_jk = get_items_in_cell(labelz_1, labelz_2)

                # now compute mean
                y_iku = np.mean(y_ik)
                y_jku = np.mean(y_jk)

                r = len(y_ik)

                denom = np.sqrt(ms_error / r)

                if len(y_ik) != len(y_jk):
                    # apply Tukey-Kramer adjustment
                    denom = np.sqrt(ms_error / 2 * (1 / len(y_ik) + 1 / len(y_jk)))

                Q = abs(y_iku - y_jku) / denom
                res_stats.append(Q)
                p = studentized_range.sf(Q, df_int, df_within)
                res_pvals.append(p)

    return TukeyHSDResult(
        comparisons=np.array(res_comp),
        statistic=np.array(res_stats),
        pvalue=np.array(res_pvals),
    )


def pairwise_welch(
    factor_a: np.typing.ArrayLike,
    factor_b: np.typing.ArrayLike,
    dependent: np.typing.ArrayLike,
    *,
    term: Literal["factor_a", "factor_b", "interaction"] = "factor_a",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> WelchResult:
    """
    Performs Pairwise Welch t-test with Holm-Bonferroni correction.

    This is a post-hoc test typically done after running a Robust Two Way ANOVA with HC3 errors.

    Parameters
    ----------
    factor_a : np.ndarray
        1D array of categorical factors for Factor A.
    factor_b : np.ndarray
        1D array of categorical factors for Factor B.
    dependent : np.ndarray
        1D array of continuous measurement observations.
    term : {'factor_a', 'factor_b', 'interaction'}, default: 'factor_a'
        Determines which main effect or interaction to analyze:
        - 'factor_a': Pairwise comparisons across Factor A levels.
        - 'factor_b': Pairwise comparisons across Factor B levels.
        - 'interaction': Pairwise comparisons across all Factor A x B cell combinations.
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle NaN inputs. Defaults to 'propagate'

    Returns
    -------
    WelchResult
        A dataclass containing all comparisons made, t-statistic, and adjusted p-values.

    References
    ----------
    .. [1] Welch, B.L. (1951). ON THE COMPARISON OF SEVERAL MEAN VALUES: AN ALTERNATIVE APPROACH. Biometrika, 38, 330-336.
    .. [2] Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics, 6, 65-70.
    """
    factor_a = np.asarray(factor_a, dtype=str)
    factor_b = np.asarray(factor_b, dtype=str)
    dependent = np.asarray(dependent, dtype=float)

    # validate
    if term not in {"factor_a", "factor_b", "interaction"}:
        raise ValueError("term must be one of {'factor_a', 'factor_b', 'interaction'}.")
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # handle nan
    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "propagate":
            return WelchResult(
                comparisons=np.array(np.nan),
                statistic=np.array(np.nan),
                pvalue=np.array(np.nan),
            )
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        # omit
        dependent = dependent[~np.isnan(dependent)]

    # depending on what term we are analzying, get the cells around it
    comparisons = []  # comparisons to analyze
    cells: dict[str, list[float]] = {}  # corresponding cells to analyze

    if term == "factor_a":
        unique_a = np.unique(factor_a)
        # get all comparisons possible and corresponding cells
        for i in unique_a:
            indecies = np.where(factor_a == i)[0]
            c = [dependent[z] for z in indecies]
            cells[i] = c
        comparisons = unique_a
    elif term == "factor_b":
        unique_b = np.unique(factor_b)
        for i in unique_b:
            indecies = np.where(factor_b == i)[0]
            c = [dependent[z] for z in indecies]
            cells[i] = c
        comparisons = unique_b
    else:
        unique_a = np.unique(factor_a)
        unique_b = np.unique(factor_b)
        b_indecies_cache: dict[str, list[int]] = {}  # just a cache to reuse
        comparisons = list(product(unique_a, unique_b))
        for i in unique_a:
            indecies_a = np.where(factor_a == i)[0]
            for z in unique_b:
                # get common indecies
                if z in b_indecies_cache:
                    indecies_b = b_indecies_cache[z]
                else:
                    # store in cache so we can reuse again on the next loop
                    indecies_b = np.where(factor_b == z)[0]
                    b_indecies_cache[z] = indecies_b
                common = np.intersect1d(indecies_a, indecies_b)
                c = [dependent[j] for j in common]
                cells[f"{i}_{z}"] = c

    # now compute everything!
    # we use the Welch t-test here
    res_stats = []
    res_pvals = []
    comp: list[str] = []
    for i in range(len(comparisons)):
        for z in range(i + 1, len(comparisons)):
            if term == "interaction":
                # comp.append(f"{i[0]}:{i[1]}-{z[0]}:{z[1]}")
                comp.append(
                    (
                        (comparisons[i][0], comparisons[i][1]),
                        (comparisons[z][0], comparisons[z][1]),
                    )
                )
            else:
                # comp.append(f"{comparisons[i]}-{comparisons[z]}")
                comp.append((comparisons[i], comparisons[z]))
            # get cells to compare
            if term == "interaction":
                cells_i = cells[f"{comparisons[i][0]}_{comparisons[i][1]}"]
                cells_z = cells[f"{comparisons[z][0]}_{comparisons[z][1]}"]
            else:
                cells_i = cells[comparisons[i]]
                cells_z = cells[comparisons[z]]

            # now do t-test
            res = ttest_ind(cells_i, cells_z, equal_var=False)
            res_stats.append(res.statistic)
            res_pvals.append(res.pvalue)

    # finally adjust
    res_pvals = np.array(res_pvals)
    m = len(res_pvals)
    order = np.argsort(res_pvals)
    sorted_p = res_pvals[order]

    adjusted_sorted = np.minimum(
        np.maximum.accumulate(sorted_p * (m - np.arange(m))), 1.0
    )

    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted

    return WelchResult(comparisons=comp, statistic=res_stats, pvalue=adjusted)


def _compute_type_2_ss(
    factor_a: np.ndarray, factor_b: np.ndarray, dependent: np.ndarray
) -> tuple[float, float, float]:
    """
    Computes Type II Sum of Squares

    Returns
    -------
    Tuples
        Sum of Squares for Factor A, Sum of Squares for Factor B, Sum of Squares for Interaction

    References
    ----------
    .. [1] Speed, F. M., et al. "Methods of Analysis of Linear Models With Unbalanced Data." Journal of the American Statistical Association, vol. 73, no. 361, Informa UK Limited, Mar. 1978, pp. 105-12, https://doi.org/10.1080/01621459.1978.10480012. Accessed 11 Nov. 2026.
    .. [2] "Types of Sum of Squares." University of Toronto, https://utstat.utoronto.ca/reid/sta442f/2009/typeSS.pdf. Accessed 11 Nov. 2026.
    """
    n = len(dependent)

    # helper function to generate dummy variables (dropping the first level to avoid collinearity)
    # categorical variables (factor_a and factor_b in this case) usually contain group labels or categories
    # we will use the helper below simply to turn these categorical labels into numeric binary indicator columns (0s and 1s)
    def get_dummies(factor):
        levels = np.unique(factor)
        if len(levels) <= 1:
            return np.empty((n, 0))
        return np.column_stack([factor == level for level in levels[1:]]).astype(float)

    # helper function to compute Residual Sum of Squares (RSS) for a given design matrix X
    def compute_rss(X):
        # build a design matrix by prepending a column of ones (representing the intercept) to the factor dummy variables.
        # add intercept column of ones
        design_matrix = (
            np.column_stack([np.ones(n), X]) if X.size > 0 else np.ones((n, 1))
        )
        # find optimal regression coefficient that minimize prediction errors for the dependent variable
        beta, residuals, _, _ = np.linalg.lstsq(design_matrix, dependent, rcond=None)

        # fallback manual calculation if lstsq returns empty residuals
        if residuals.size == 0:
            # compute model predictions and sums the squared residuals (errors) between the actual and predicted values
            pred = design_matrix @ beta
            return np.sum((dependent - pred) ** 2)
        return residuals[0]

    # generate dummy matrices for factors A and B
    X_a = get_dummies(factor_a)
    X_b = get_dummies(factor_b)

    # generate interaction dummy variables (pairwise products of A and B dummies)
    interaction_cols = []
    for i in range(X_a.shape[1]):
        for j in range(X_b.shape[1]):
            interaction_cols.append(X_a[:, i] * X_b[:, j])
    X_ab = np.column_stack(interaction_cols) if interaction_cols else np.empty((n, 0))

    # helper to combine main effect matrices safely
    # all this does in short is literally taking multiple design matrices and joining them side-by-side horizontally
    # this will help us compute R(A, B) and R(A, B, AB) which is critical to computing Type II Sum of Squares
    def combine_matrices(*matrices):
        valid = [m for m in matrices if m.size > 0]
        return np.column_stack(valid) if valid else np.empty((n, 0))

    # compute residual sum of squares for each
    r_a = compute_rss(X_a)  # R(A)
    r_b = compute_rss(X_b)  # R(B)
    r_ab = compute_rss(combine_matrices(X_a, X_b))  # R(A, B)
    r_full = compute_rss(combine_matrices(X_a, X_b, X_ab))  # R(A, B, AB)

    # now apply the type II formulas:
    # SS(A | B) = R(B) - R(A, B)
    ss_a = r_b - r_ab

    # SS(B | A) = R(A) - R(A, B)
    ss_b = r_a - r_ab

    # SS(AB | A, B) = R(A, B) - R(A, B, AB)
    ss_interaction = r_ab - r_full

    return ss_a, ss_b, ss_interaction


def scheirer(
    factor_a: np.ndarray,
    factor_b: np.ndarray,
    dependent: np.ndarray,
    *,
    tie_correction: bool = True,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> ScheirerResult:
    """
    Performs the Scheirer-Ray-Hare Test (non-parametric two-way ANOVA).

    Parameters
    ----------
    factor_a : np.ndarray
        1D array of categorical factors for Factor A.
    factor_b : np.ndarray
        1D array of categorical factors for Factor B.
    dependent : np.ndarray
        1D array of continuous measurement observations.
    tie_correction : bool
        Whether to adjust the H-Statistic when ties exist. Defaults to True
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle NaNs in input arrays.

    Returns
    -------
    ScheirerResult
        Data Class containing H-statistics, p-values, and degrees of freedom.

    See Also
    --------
    art : Aligned Rank Transform (ART) ANOVA for non-parametric factorial designs.

    Note
    ----
    The formula to compute the H-Statistic here is:

    $H_effect = SS_effect / MS_total$

    where *SS_effect* is the Sum of Squares (depending on what H-Statistic you are trying to compute, it may be for Factor A, Factor B, or Interaction) and *MS_total* is:

    $MS_total = {N(N^2 - 1)}/{12(N - 1)}$

    where *N* is the length of the `dependent` variable in short.

    This function strictly implements Type II Sum of Squares to compute *SS_effect.*

    A word on how degrees of freedom are computed for the interaction (see `ScheirerResult.df_interaction`). Degrees of freedom are computed from the number of filled cells in the design. For a complete factorial design this equals (a-1)(b-1); for incomplete designs it is reduced by the number of empty cells, reflecting the reduced rank of the interaction subspace. This makes it so the function can handle balanced and unbalanced designs as needed.

    Finally, the `tie_correction` is used in case there are any ties in the `dependent` variable. This is technically *optional* to apply. When applied, it will impact the H-Statistics (all three) and the resulting p-values. Note that this does not remove the ties from the data itself it merely accounts for their presence in the variance calculation. It is generally recommended though to apply corrections in case of ties, hence leaving the default value of this parameter to `True`.

    References
    ----------
    .. [1] Scheirer, C. J., Ray, W. S., & Hare, N. (1976). The Analysis of Ranked Data Derived from Completely Randomized Factorial Designs. Biometrics, 32(2), 429-434. https://doi.org/10.2307/2529511
    .. [2] Kiral, Işik. (2019). Re: Effect Size Calculation for Scheirer-Ray-Hare Test. Retrieved from: https://www.researchgate.net/post/Effect_Size_Calculation_for_Scheirer-Ray-Hare_Test/5d659902a7cbafda215a9eb1/citation/download.
    """
    factor_a = np.asarray(factor_a)
    factor_b = np.asarray(factor_b)
    dependent = np.asarray(dependent, dtype=float)

    if factor_a.ndim != 1 or factor_b.ndim != 1 or dependent.ndim != 1:
        raise ValueError("All input arrays must be one-dimensional.")
    if not (factor_a.shape == factor_b.shape == dependent.shape):
        raise ValueError("All input arrays must have the same shape.")

    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'.")

    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        if nan_policy == "propagate":
            return ScheirerResult(
                statistic_a=np.nan,
                pvalue_a=np.nan,
                df_a=np.nan,
                statistic_b=np.nan,
                pvalue_b=np.nan,
                df_b=np.nan,
                statistic_interaction=np.nan,
                pvalue_interaction=np.nan,
                df_interaction=np.nan,
            )
        factor_a = factor_a[~np.isnan(factor_a)]
        factor_b = factor_b[~np.isnan(factor_b)]
        dependent = dependent[~np.isnan(dependent)]

    N = len(dependent)
    if N == 0:
        raise ValueError("At least one complete observation is required.")

    # first rank data
    ranks = rankdata(dependent, method="average")
    # get sum of squares for factors and interaction
    ss_a, ss_b, ss_interaction = _compute_type_2_ss(factor_a, factor_b, ranks)
    # compute degrees of freedom
    levels_a = np.unique(factor_a)
    levels_b = np.unique(factor_b)
    # df for Factor A = a - 1
    df_a = len(levels_a) - 1
    # df for Factor B = b - 1
    df_b = len(levels_b) - 1
    # df for interaction = (a - 1) * (b - 1)
    # HOWEVER, we will compute the degrees of freedom based on number of filled cells in the design
    n_cells = len(set(zip(factor_a, factor_b)))
    df_int = n_cells - len(levels_a) - len(levels_b) + 1
    # compute total mean square
    # ss_total = np.sum((ranks - np.mean(ranks)) ** 2)
    # ms_total = N(N^2 - 1) / 12(N - 1)
    ms_total = (N * (N**2 - 1)) / (12 * (N - 1))

    # now compute statistic (h = ss_effect / mean_total)
    h_a = ss_a / ms_total
    h_b = ss_b / ms_total
    h_int = ss_interaction / ms_total

    # check if we need to adjust
    if tie_correction:
        # check if dependent variable has any ties
        ties: dict[float, int] = defaultdict(int)
        for i in ranks:
            ties[i] += 1

        # now go over ties
        t: float = 0.0
        for i, value in ties.items():
            if value > 1:
                t += value**3 - value

        if t > 0:
            D = 1.0 - (t / (N**3 - N))
            # adjust each statistic based on this correction
            h_a = h_a / D
            h_b = h_b / D
            h_int = h_int / D

    p_a = chi2.sf(h_a, df_a)
    p_b = chi2.sf(h_b, df_b)
    p_int = chi2.sf(h_int, df_int)

    # Format returns
    # statistics = np.array([h_a, h_b, h_int])
    # pvalues = np.array([p_a, p_b, p_int])
    # dfs = np.array([df_a, df_b, df_int])
    # sum_squares = np.array([ss_a, ss_b, ss_interaction, ss_total])

    return ScheirerResult(
        statistic_a=h_a,
        pvalue_a=p_a,
        df_a=df_a,
        statistic_b=h_b,
        pvalue_b=p_b,
        df_b=df_b,
        statistic_interaction=h_int,
        pvalue_interaction=p_int,
        df_interaction=df_int,
    )


#############
# ART ANOVA #
#############


def _compute_type_iii_ss_and_f(
    factor_a: np.ndarray, factor_b: np.ndarray, y: np.ndarray
) -> dict:
    """
    Computes Type III Sum of Squares, F-statistics, and p-values
    for a two-way factorial design using pure NumPy linear algebra.

    References
    ----------
    .. [1] Speed, F. M., et al. "Methods of Analysis of Linear Models With Unbalanced Data." Journal of the American Statistical Association, vol. 73, no. 361, Informa UK Limited, Mar. 1978, pp. 105-12, https://doi.org/10.1080/01621459.1978.10480012. Accessed 11 Nov. 2026.
    .. [2] "Types of Sum of Squares." University of Toronto, https://utstat.utoronto.ca/reid/sta442f/2009/typeSS.pdf. Accessed 11 Nov. 2026.
    """
    # get unique levels and encode factors as zero-indexed integers
    a_levels, a_inv = np.unique(factor_a, return_inverse=True)
    b_levels, b_inv = np.unique(factor_b, return_inverse=True)

    n_a = len(a_levels)
    n_b = len(b_levels)
    n_total = len(y)

    # build Sum-to-Zero (Effect Coding) Contrasts
    # this is required for Type III sums of squares so effects are orthogonal.
    def get_sum_contrasts(indices, levels):
        contrasts = np.zeros((len(indices), max(levels - 1, 0)))
        for lvl in range(levels - 1):
            contrasts[indices == lvl, lvl] = 1.0
        if levels > 1:
            contrasts[indices == levels - 1, :] = -1.0
        return contrasts

    a_mat = get_sum_contrasts(a_inv, n_a)
    b_mat = get_sum_contrasts(b_inv, n_b)

    # tntercept
    intercept = np.ones((n_total, 1))

    # interaction columns (element-wise multiplication of contrasts)
    interaction_cols = [
        (a_mat[:, i] * b_mat[:, j])[:, None]
        for i in range(a_mat.shape[1])
        for j in range(b_mat.shape[1])
    ]

    # construct full design matrix X: [Intercept, Factor A, Factor B, Interaction]
    if interaction_cols:
        X = np.column_stack((intercept, a_mat, b_mat, *interaction_cols))
    else:
        X = np.column_stack((intercept, a_mat, b_mat))

    # define slice blocks for each term in the design matrix
    a_width = max(n_a - 1, 0)
    b_width = max(n_b - 1, 0)

    # intercept_slice = slice(0, 1)
    a_slice = slice(1, 1 + a_width)
    b_slice = slice(1 + a_width, 1 + a_width + b_width)
    ab_slice = slice(1 + a_width + b_width, X.shape[1])

    # fit Ordinary Least Squares (OLS) regression: beta = (X^T X)^(-1) X^T y
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    ss_error = float(np.sum(residuals**2))

    # Degrees of Freedom
    df_error = n_total - rank
    if df_error <= 0:
        raise ValueError("Error degrees of freedom must be positive.")

    # parameter covariance matrix
    # using pseudoinverse (pinv) handles rank deficiency safely
    xtx_inv = np.linalg.pinv(X.T @ X)

    # compute Type III Sum of Squares via Wald Quadratic Form for each block
    def calc_ss_block(block_slice):
        coefs = beta[block_slice]
        if len(coefs) == 0:
            return 0.0, 0
        cov_block = xtx_inv[block_slice, block_slice]
        # Type III SS = beta_block^T * inv(cov_block) * beta_block
        ss = float(coefs @ np.linalg.pinv(cov_block) @ coefs)
        df = np.linalg.matrix_rank(X[:, block_slice])
        return max(ss, 0.0), df

    ss_a, df_a = calc_ss_block(a_slice)
    ss_b, df_b = calc_ss_block(b_slice)
    ss_ab, df_ab = calc_ss_block(ab_slice)

    # calculate Mean Squares, F-statistics, and p-values
    ms_error = ss_error / df_error

    def make_result(ss, df):
        if df == 0 or ms_error == 0:
            return 0.0, 1.0, 0.0, float(df_error)
        ms_effect = ss / df
        f_stat = ms_effect / ms_error
        p_val = float(f.sf(f_stat, df, df_error))
        return f_stat, p_val, float(df), float(df_error)

    f_a, p_a, da, de = make_result(ss_a, df_a)
    f_b, p_b, db, _ = make_result(ss_b, df_b)
    f_ab, p_ab, dab, _ = make_result(ss_ab, df_ab)

    return {
        "factor_a": {"statistic": f_a, "pvalue": p_a, "df_num": da, "df_den": de},
        "factor_b": {"statistic": f_b, "pvalue": p_b, "df_num": db, "df_den": de},
        "interaction": {"statistic": f_ab, "pvalue": p_ab, "df_num": dab, "df_den": de},
    }


def art(
    factor_a: np.typing.ArrayLike,
    factor_b: np.typing.ArrayLike,
    dependent: np.typing.ArrayLike,
    *,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> ARTResult:
    """
    Aligned Rank Transform (ART) Two-Way ANOVA for non-parametric factorial data.

    Parameters
    ----------
    factor_a : array_like
        First categorical independent variable.
    factor_b : array_like
        Second categorical independent variable.
    dependent : array_like
        Continuous response/dependent variable.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle NaNs in input data. Defaults to 'propagate'.

    Returns
    -------
    ARTResult
        Object containing F-statistics, p-values, and degrees of freedom for
        Factor A, Factor B, and their Interaction.

    Notes
    -----
    ART ANOVA is a non-parametric function that can be run when assumptions for the Two Way ANOVA are not met.

    It is also very important to note how this function implements strictly the **Type III Sum of Squares** method for calculating the F-statistics and p-values for each factor and interaction. This makes it so the function can even handle unbalanced designs (i.e., unequal sample sizes across groups).

    This function *does not* perform any post-hoc tests, it only computes the statistics and p-values. Typically, to perform a post-hoc test after ART ANOVA (assuming your results are significant of course), you would use the ART-C Framework.

    **Extremely important note:** AnovaPy implements the Aligned Rank Transform using native NumPy `np.float64` precision. In rare edge cases involving extreme floating-point boundaries (for instance near the 15th decimal place), tie-breaking behavior may differ slightly from R's ARTool due to R's internal formatting/rounding differences. This results in higher mathematical precision in AnovaPy.

    References
    ----------
    .. [1] Wobbrock, J. O., Findlater, L., Gergle, D., & Higgins, J. J. (2011). The aligned rank transform for nonparametric factorial analyses using only ANOVA procedures. Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, 143-146. https://doi.org/10.1145/1978942.1978963
    """
    factor_a = np.asarray(factor_a)
    factor_b = np.asarray(factor_b)
    dependent = np.asarray(dependent)

    # check nan values
    if factor_a.ndim != 1 or factor_b.ndim != 1 or dependent.ndim != 1:
        raise ValueError("All input arrays must be one-dimensional.")
    if not (factor_a.shape == factor_b.shape == dependent.shape):
        raise ValueError("All input arrays must have the same shape.")
    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'.")

    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "propagate":
            return ARTResult(
                statistic_a=np.nan,
                pvalue_a=np.nan,
                df_num_a=np.nan,
                df_den_a=np.nan,
                statistic_b=np.nan,
                pvalue_b=np.nan,
                df_num_b=np.nan,
                df_den_b=np.nan,
                statistic_interaction=np.nan,
                pvalue_interaction=np.nan,
                df_num_interaction=np.nan,
                df_den_interaction=np.nan,
            )
        if nan_policy == "raise":
            raise ValueError("The input contains nan values")
        dependent = dependent[~np.isnan(dependent)]
    if len(dependent) == 0:
        raise ValueError("At least one complete observation is required.")

    # compute residuals
    # residual = Y - cell_mean
    # compute cell mean for each
    # stack factors into a 2D array of "keys"
    keys = np.column_stack([factor_a, factor_b])

    # find unique cells and get group indecies
    _unique_cells, inverse = np.unique(keys, axis=0, return_inverse=True)

    # compute sums and counts per group
    sums = np.bincount(inverse, weights=dependent)
    counts = np.bincount(inverse)

    # now compute the cell mean by simply dividing the two
    cell_means_unique = sums / counts

    # now just map out the indecies so we can easily compute residuals
    cell_means = cell_means_unique[inverse]

    # now compute residuals
    residuals = dependent - cell_means

    # now compute estimated effects
    grand_mean = np.mean(dependent)

    # for one way effect (so just the factors), the formula is effect = A_i - grand_mean
    # for two way effect (interaction), the formula is a_i*b_i - a_i - b_i + grand_mean
    _a_levels, a_inv = np.unique(factor_a, return_inverse=True)
    _b_levels, b_inv = np.unique(factor_b, return_inverse=True)

    # a_means = np.array([np.mean(dependent[a_inv == i]) for i in range(len(factor_a))])
    # b_means = np.array([np.mean(dependent[b_inv == j]) for j in range(len(factor_b))])
    a_means = np.bincount(a_inv, weights=dependent) / np.bincount(a_inv)
    b_means = np.bincount(b_inv, weights=dependent) / np.bincount(b_inv)

    ab_means = np.zeros((len(factor_a), len(factor_b)))
    for i in range(len(factor_a)):
        for j in range(len(factor_b)):
            mask = (a_inv == i) & (b_inv == j)
            if np.any(mask):
                ab_means[i, j] = np.mean(dependent[mask])

    # transform back to proper indecies
    mu_a = a_means[a_inv]
    mu_b = b_means[b_inv]
    mu_ab = ab_means[a_inv, b_inv]

    effect_a = mu_a - grand_mean
    effect_b = mu_b - grand_mean
    effect_ab = mu_ab - mu_a - mu_b + grand_mean

    # compute aligned reponse Y'
    Y_a = residuals + effect_a
    Y_b = residuals + effect_b
    Y_ab = residuals + effect_ab

    # assign averged ranks Y''
    # the smallest Y' receives rank 1
    # the next smallest gets rank 2 and so on
    # if there is a tie, break it by taking the average
    ranked_a = rankdata(Y_a, method="average")
    ranked_b = rankdata(Y_b, method="average")
    ranked_ab = rankdata(Y_ab, method="average")

    # now perform full factorial ANOVA on Y''
    res_a = _compute_type_iii_ss_and_f(factor_a, factor_b, ranked_a)["factor_a"]
    res_b = _compute_type_iii_ss_and_f(factor_a, factor_b, ranked_b)["factor_b"]
    res_ab = _compute_type_iii_ss_and_f(factor_a, factor_b, ranked_ab)["interaction"]

    return ARTResult(
        statistic_a=res_a["statistic"],
        pvalue_a=res_a["pvalue"],
        df_num_a=res_a["df_num"],
        df_den_a=res_a["df_den"],
        statistic_b=res_b["statistic"],
        pvalue_b=res_b["pvalue"],
        df_num_b=res_b["df_num"],
        df_den_b=res_b["df_den"],
        statistic_interaction=res_ab["statistic"],
        pvalue_interaction=res_ab["pvalue"],
        df_num_interaction=res_ab["df_num"],
        df_den_interaction=res_ab["df_den"],
    )


def artc(
    factor_a: np.typing.ArrayLike,
    factor_b: np.typing.ArrayLike,
    dependent: np.typing.ArrayLike,
    *,
    term: Literal["factor_a", "factor_b", "interaction"] = "factor_a",
    interaction: bool = False,
    adjust: Literal["holm", "bonferroni", "none"] = "holm",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> ARTContrastResult:
    """
    Aligned Rank Transform Contrast (ART-C) post-hoc pairwise tests for ART ANOVA.

    Parameters
    ----------
    factor_a : array_like
        First categorical independent variable.
    factor_b : array_like
        Second categorical independent variable.
    dependent : array_like
        Continuous response/dependent variable.
    term : {'factor_a', 'factor_b', 'interaction'}
        The effect/factor for which pairwise contrasts are tested. Default is 'factor_a'.
    interaction : bool, optional
        Whether to test differences of differences on data that has been aligned-and-ranked using the original ART method. Default is False. **WARNING: this functionality has not yet been implemented and when set to `True` a `NotImplementedError` will be thrown!**
    adjust : {'holm', 'bonferroni', 'none'}, optional
        P-value adjustment method for multiple comparisons. Default is 'holm'.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle NaNs in input data. Default is 'propagate'.

    Returns
    -------
    ARTContrastResult
        Calculated pairwise comparison statistics, t-values, and adjusted p-values.

    Notes
    -----
    This function, just like the `art` function, has been desinged to strictly handle Two Way designs.

    To test for differences within a single factor (Factor A or Factor B), just set the `term` parameter to either 'factor_a' or 'factor_b'.

    A word on the `interaction` parameter: this is used to check for **differences of differences** (e.g., is the difference between levels of A different at level 1 of B compared to level 2 of B?). It defaults to `False` automatically. Though when set to `True`, the `term` parameter must be set to `'interaction'` as well. This is because the interaction term is the only one that can be used to test for differences of differences.
    Note that running this function with `interaction=True` and `term='interaction'` will NOT get the same result as running with `interaction=False` and `term='interaction'`. The former will only test for differences of differences (Is the A1-A2 gap at B1 different from the A1-A2 gap at B2?), while the later tests for pairwise cell differences (e.g., "Is the mean at A1:B1 different from A1:B2?").

    **WARNING:** `interaction=True` has NOT been implemented yet and when set to `True` a `NotImplementedError` will be thrown!

    **Extremely important note:** AnovaPy implements the Aligned Rank Transform Contrast post-hoc using native NumPy `np.float64` precision. In rare edge cases involving extreme floating-point boundaries (for instance near the 15th decimal place), tie-breaking behavior may differ slightly from R's ARTool due to R's internal formatting/rounding differences. This results in higher mathematical precision in AnovaPy.

    References
    ----------
    .. [1] Elkin, L. A., Kay, M., Higgins, J. J., & Wobbrock, J. O. (2021). An aligned rank transform procedure for multifactor contrast tests. The 34th Annual ACM Symposium on User Interface Software and Technology. https://doi.org/10.1145/3472749.3474784
    """
    # make factor_a and factor_b string so concatenation is simpler
    factor_a = np.asarray(factor_a)
    factor_b = np.asarray(factor_b)
    dependent = np.asarray(dependent)

    # check nan values
    if factor_a.ndim != 1 or factor_b.ndim != 1 or dependent.ndim != 1:
        raise ValueError("All input arrays must be one-dimensional.")
    if not (factor_a.shape == factor_b.shape == dependent.shape):
        raise ValueError("All input arrays must have the same shape.")
    if nan_policy not in ("propagate", "omit", "raise"):
        raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'.")

    has_nan = np.isnan(dependent).any()
    if has_nan:
        if nan_policy == "propagate":
            return ARTContrastResult(comparisons=[], statistic=[], pvalue=[], df=[])
        if nan_policy == "raise":
            raise ValueError("The input contains nan values")
        dependent = dependent[~np.isnan(dependent)]
    if len(dependent) == 0:
        raise ValueError("At least one complete observation is required.")

    # compute grand mean ahead of time
    grand_mean = np.mean(dependent)

    # what exactly are we checking?
    if interaction:
        # we literally just use the original ART method
        if term != "interaction":
            warnings.warn(
                f"term='interaction' was expected but got term='{term}'. For interaction contrasts, term should be 'interaction'. This was automatically corrected."
            )

        # compute mean for A and B
        _unique_a, inv_a = np.unique(factor_a, return_inverse=True)
        _unique_b, inv_b = np.unique(factor_b, return_inverse=True)
        mean_a = (np.bincount(inv_a, weights=dependent) / np.bincount(inv_a))[inv_a]
        mean_b = (np.bincount(inv_b, weights=dependent) / np.bincount(inv_b))[inv_b]

        # align
        y_aligned = dependent - mean_a - mean_b + grand_mean
        _y_ranked = rankdata(y_aligned, method="average")

        raise NotImplementedError(
            "Interaction constrasts (`interaction=True`) are not yet implemented!"
        )

    # first prepare the data
    # concatenate factor_a and factor_b where for any response Y (dependent) for which A has level factor_a_i and B has level factor_b_j, and AB has level AB_{ij}
    AB = np.char.add(factor_a, np.char.add("_", factor_b))

    # compute aligned response Y'
    # compute cell means for each unique group in AB
    _unique_AB, inv_AB = np.unique(AB, return_inverse=True)
    sums = np.bincount(inv_AB, weights=dependent)
    counts = np.bincount(inv_AB)

    # now compute the cell mean by simply dividing the two
    cell_means_unique = sums / counts

    # now just map out the indecies
    cell_means = cell_means_unique[inv_AB]

    # for single factor, Y' = Y_{ijk} - mean(A_iB_j) + mean(A_i|B_i) - grand_mean
    # for two factor (interaction), Y' = Y_{ijk} - grand_mean

    unique_a, inv_a = np.unique(factor_a, return_inverse=True)
    unique_b, inv_b = np.unique(factor_b, return_inverse=True)
    mean_a = (np.bincount(inv_a, weights=dependent) / np.bincount(inv_a))[inv_a]
    mean_b = (np.bincount(inv_b, weights=dependent) / np.bincount(inv_b))[inv_b]

    y_prime = 0.0  # to be computed
    if term == "factor_a":
        y_prime = dependent - cell_means + mean_a - grand_mean
    elif term == "factor_b":
        y_prime = dependent - cell_means + mean_b - grand_mean
    else:  # term == 'interaction'
        y_prime = dependent - grand_mean

    # compute ranked response Y''
    y_double_prime = rankdata(y_prime, method="average")

    # conduct constrat tests on Y''
    # perform pairwise comparisons between all unique levels of AB using Y'' as the dependent variable
    levels_a = np.unique(factor_a)
    levels_b = np.unique(factor_b)
    nA, nB = len(levels_a), len(levels_b)

    a_idx = np.searchsorted(levels_a, factor_a)
    b_idx = np.searchsorted(levels_b, factor_b)

    # build sum-to-zero design matrix X for full factorial model (A * B)
    X_cols = [np.ones(len(dependent))]

    for i in range(nA - 1):
        col = np.where(a_idx == i, 1.0, 0.0)
        col[a_idx == nA - 1] = -1.0
        X_cols.append(col)

    for j in range(nB - 1):
        col = np.where(b_idx == j, 1.0, 0.0)
        col[b_idx == nB - 1] = -1.0
        X_cols.append(col)

    for i in range(nA - 1):
        for j in range(nB - 1):
            col_a = X_cols[1 + i]
            col_b = X_cols[nA + j]
            X_cols.append(col_a * col_b)

    X = np.column_stack(X_cols)

    # fit OLS: beta = (X^T X)^-1 X^T y
    beta, _, rank, _ = np.linalg.lstsq(X, y_double_prime, rcond=None)

    # compute residual variance and degrees of freedom
    residuals = y_double_prime - X @ beta
    N = len(y_double_prime)
    df_res = N - rank
    mse = np.sum(residuals**2) / df_res if df_res > 0 else 0.0

    # covariance matrix of coefficients
    XtX_inv = np.linalg.pinv(X.T @ X)
    cov_beta = mse * XtX_inv

    interaction_groups: dict[
        str, tuple[Any, Any]
    ] = {}  # specifically for interaction only

    # define contrast helper based on term
    if term == "factor_a":
        contrast_groups = levels_a

        def get_contrast_vector(g1, g2):
            idx1 = np.where(levels_a == g1)[0][0]
            idx2 = np.where(levels_a == g2)[0][0]
            c = np.zeros(len(beta))
            # sum-to-zero contrast mapping for factor A
            if idx1 < nA - 1:
                c[1 + idx1] = 1.0
            else:
                for k in range(nA - 1):
                    c[1 + k] = -1.0
            if idx2 < nA - 1:
                c[1 + idx2] = -1.0
            else:
                for k in range(nA - 1):
                    c[1 + k] += 1.0
            return c
    elif term == "factor_b":
        contrast_groups = levels_b
        b_start = nA

        def get_contrast_vector(g1, g2):
            idx1 = np.where(levels_b == g1)[0][0]
            idx2 = np.where(levels_b == g2)[0][0]
            c = np.zeros(len(beta))
            if idx1 < nB - 1:
                c[b_start + idx1] = 1.0
            else:
                for k in range(nB - 1):
                    c[b_start + k] = -1.0
            if idx2 < nB - 1:
                c[b_start + idx2] = -1.0
            else:
                for k in range(nB - 1):
                    c[b_start + k] += 1.0
            return c
    else:  # interaction (AB levels)
        contrast_groups = np.unique(AB)

        # set up terms
        for i in unique_a:
            for z in unique_b:
                key = f"{i}_{z}"
                interaction_groups[key] = (i, z)

        def get_contrast_vector(g1, g2):
            # find the factor_a and factor_b levels for group strings g1 and g2 safely
            mask1 = AB == g1
            a1, b1 = factor_a[mask1][0], factor_b[mask1][0]

            mask2 = AB == g2
            a2, b2 = factor_a[mask2][0], factor_b[mask2][0]

            i1 = np.where(levels_a == a1)[0][0]
            j1 = np.where(levels_b == b1)[0][0]
            i2 = np.where(levels_a == a2)[0][0]
            j2 = np.where(levels_b == b2)[0][0]

            def get_row(i, j):
                row = np.zeros(len(beta))
                row[0] = 1.0  # intercept

                # Factor A parameters
                for ki in range(nA - 1):
                    if i == ki:
                        row[1 + ki] = 1.0
                    elif i == nA - 1:
                        row[1 + ki] = -1.0

                # Factor B parameters
                for kj in range(nB - 1):
                    if j == kj:
                        row[nA + kj] = 1.0
                    elif j == nB - 1:
                        row[nA + kj] = -1.0

                # Interaction parameters (products of A and B design columns)
                int_idx = nA + nB - 1
                for ki in range(nA - 1):
                    for kj in range(nB - 1):
                        val_a = row[1 + ki]
                        val_b = row[nA + kj]
                        row[int_idx] = val_a * val_b
                        int_idx += 1
                return row

            return get_row(i1, j1) - get_row(i2, j2)

    comparisons = list(combinations(contrast_groups, 2))

    res_comparisons = []
    res_df = []
    res_stats = []
    res_pvals = []

    for group1, group2 in comparisons:
        if term == "interaction":
            res_comparisons.append(
                (interaction_groups[group1], interaction_groups[group2])
            )
        else:
            res_comparisons.append((group1, group2))
        c = get_contrast_vector(group1, group2)

        # estimate = c^T * beta
        estimate = np.dot(c, beta)

        # standard error = sqrt(c^T * Cov(beta) * c)
        se = np.sqrt(np.dot(c, np.dot(cov_beta, c)))

        # t-statistic and model residual degrees of freedom
        t_stat = estimate / se if se > 0 else 0.0
        df = df_res

        unadj_p_value = 2 * (1.0 - t.cdf(np.abs(t_stat), df))

        res_df.append(df)
        res_stats.append(t_stat)
        res_pvals.append(unadj_p_value)

    # adjust p-values
    n_tests = len(res_pvals)

    res_pvals = np.array(res_pvals)

    if adjust == "bonferroni":
        adjusted_p = np.minimum(1.0, res_pvals * n_tests)
    elif adjust == "holm":
        sorted_idx = np.argsort(res_pvals)
        sorted_p = res_pvals[sorted_idx]
        adjusted_p = np.empty_like(res_pvals)

        current_min = 0.0
        for i, idx in enumerate(sorted_idx):
            val = sorted_p[i] * (n_tests - i)
            current_min = max(current_min, val)
            adjusted_p[idx] = min(1.0, current_min)
    else:  # adjust == 'none'
        adjusted_p = res_pvals

    return ARTContrastResult(
        comparisons=res_comparisons, statistic=res_stats, pvalue=adjusted_p, df=res_df
    )


#######################
# Data Transformation #
#######################


def transform_dependent(
    dependent: np.ndarray,
    *,
    method: Literal["log", "square_root", "reciprocal", "box_cox"] = "log",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> np.ndarray:
    """
    Transforms a continuous dependent variable to help satisfy ANOVA assumptions.

    Parameters
    ----------
    dependent : np.ndarray
        1D array of continuous numeric values.
    method : {'log', 'square_root', 'reciprocal', 'box_cox'}
        Transformation method to apply. Typically 'log' is used for highly (positively) skewed data. The square root is usually used on moderately positively skewed data. The reciprocal method is typically used for extreme positive skew (used when you want to look at the rate of something rather than the total time). The Box-Cox method can be used when normality and homogeneity of variance are both violated and when you don't want to guess. It tests a range of power transformations to mathematically find the perfect fit. The log method seems to be used most often, hence it remains the default.
    nan_policy : {'propagate', 'omit', 'raise'}, default='propagate'
        Defines how to handle NaN values:
        - 'propagate': returns NaN values in output.
        - 'omit': ignores NaNs when performing calculation (drops them before transform).
        - 'raise': throws an error if NaNs are present.

    Returns
    -------
    np.ndarray
        Transformed dependent variable.

    References
    ----------
    .. [1] Box, G. E. P., and D. R. Cox. "An Analysis of Transformations." Journal of the Royal Statistical Society Series B: Statistical Methodology, vol. 26, no. 2, Oxford University Press (OUP), July 1964, pp. 211-43, https://doi.org/10.1111/j.2517-6161.1964.tb00553.x.
    .. [2] Jaeger TF. Categorical Data Analysis: Away from ANOVAs (transformation or not) and towards Logit Mixed Models. J Mem Lang. 2008 Nov;59(4):434-446. doi: 10.1016/j.jml.2007.11.007. PMID: 19884961; PMCID: PMC2613284.
    .. [3] Lee DK. Data transformation: a focus on the interpretation. Korean J Anesthesiol. 2020 Dec;73(6):503-508. doi: 10.4097/kja.20137. Epub 2020 Nov 20. PMID: 33271009; PMCID: PMC7714623.
    """
    dep = np.asarray(dependent, dtype=float)

    # Handle NaN Policy
    has_nan = np.isnan(dep).any()
    if has_nan:
        if nan_policy == "raise":
            raise ValueError("Input array contains NaN values.")
        elif nan_policy == "omit":
            dep = dep[~np.isnan(dep)]

    # Apply Transformations
    if method == "log":
        if np.any(dep <= 0):
            raise ValueError(
                "Log transformation requires strictly positive values (> 0)."
            )
        transformed = np.log(dep)

    elif method == "square_root":
        if np.any(dep < 0):
            raise ValueError(
                "Square root transformation requires non-negative values (>= 0)."
            )
        transformed = np.sqrt(dep)

    elif method == "reciprocal":
        if np.any(dep == 0):
            raise ValueError("Reciprocal transformation cannot handle zero values.")
        transformed = 1.0 / dep

    elif method == "box_cox":
        if np.any(dep <= 0):
            raise ValueError(
                "Box-Cox transformation requires strictly positive values (> 0)."
            )
        # stats.boxcox returns (transformed_data, lambda_param); we only keep transformed data
        transformed, _ = boxcox(dep)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'log', 'square_root', 'reciprocal', 'box_cox'."
        )

    return transformed


def transform_and_run_twoway(
    factor_a: np.ndarray,
    factor_b: np.ndarray,
    dependent: np.ndarray,
    *,
    method: Literal["log", "square_root", "reciprocal", "box_cox"] = "log",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> TwoWayAnovaResult:
    """
    Transforms the dependent variable and automatically executes the two-way ANOVA function.

    Parameters
    ----------
    factor_a : np.ndarray
        Categorical array for Factor A.
    factor_b : np.ndarray
        Categorical array for Factor B.
    dependent : np.ndarray
        1D array of continuous numeric values.
    method : {'log', 'square_root', 'reciprocal', 'box_cox'}
        Transformation method. Default to 'log'.
    nan_policy : {'propagate', 'omit', 'raise'}, default='propagate'
        Handling strategy for NaNs.

    Returns
    -------
    TwoWayAnovaResult
        The output of the two way ANOVA function after the data transformation

    Examples
    --------
    Note that this function is the equivalent of calling `transform_dependent` and then `f_twoway` as shown below:

    >>> from anovapy.twoway import f_twoway, transform_dependent, transform_and_run_twoway
    >>> import numpy as np
    >>> factor_a = np.repeat([0, 1], 20) # note that 0 in this case represents 'Group1' and 1 represents 'Group2' -- regardless you can put strings instead of numbers here, it doesn't matter
    >>> factor_b = np.tile(np.repeat([0, 1], 10), 2) # 0 represents 'LevelA' and 1 represents 'LevelB'
    >>> dependent = np.random.exponential(scale=2.0, size=40) + 5.0 # just give right skew
    >>> dependent_transformed = transform_dependent(dependent=dependent, method='log')
    >>> two_way_res = f_twoway(factor_a=factor_a, factor_b=factor_b, dependent=dependent_transformed)
    >>> two_way_res2 = transform_and_run_twoway(factor_a=factor_a, factor_b=factor_b, dependent=dependent, method='log')
    >>> # two_way_res and two_way_res2 are exactly the same
    """
    transformed_dep = transform_dependent(
        dependent, method=method, nan_policy=nan_policy
    )

    # Pass original categorical factors with the newly transformed dependent variable
    return f_twoway(factor_a, factor_b, transformed_dep, nan_policy=nan_policy)
