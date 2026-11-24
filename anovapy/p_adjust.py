# just for adjusting p-values in post-hoc tests
# this can also be used to adjust p-values more efficiently
# for example, if you are running a post-hoc test and for whatever reason want to see how different p-values appear under different adjustment methods,
# you can first compute the unadjusted p-values and then run some of the methods directly on those
# which lowers the computation time because you don't have to compute the statistic again!

from typing import Literal

import numpy as np

PValueAdjustmentMethod = Literal[
    "bonferroni",
    "holm",
    "sidak",
    "benjamini",
    "benjamini_yekutieli",
    "hochberg",
    "none",
]


def bonferroni(pvalue: np.typing.ArrayLike) -> np.ndarray:
    """
    Adjusts p-values using the Bonferroni method

    Parameters
    ----------
    pvalue : array_like
        A 1D array of p-values

    Returns
    -------
    All adjusted p-values
    """
    pvalue = np.asarray(pvalue)
    return np.minimum(pvalue * len(pvalue), 1)


def holm(pvalue: np.typing.ArrayLike) -> np.ndarray:
    """
    Adjusts p-values with the Holm-Bonferroni method.

    Parameters
    ----------
    pvalue : array_like
        A 1D array of p-values

    Returns
    -------
    All adjusted p-values
    """
    pvalue = np.asarray(pvalue)
    order = np.argsort(pvalue)
    sorted_p = pvalue[order]

    m = len(pvalue)

    adjusted_sorted = np.minimum(
        np.maximum.accumulate(sorted_p * (m - np.arange(m))), 1.0
    )

    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted

    return adjusted


def sidak(pvalue: np.typing.ArrayLike) -> np.ndarray:
    """
    Adjusts p-values with the Sidak method.

    Parameters
    ----------
    pvalue : array_like
        A 1D array of p-values

    Returns
    -------
    All adjusted p-values
    """
    pvalue = np.asarray(pvalue)
    m = len(pvalue)
    return 1 - (1 - pvalue) ** m


def benjamini_hochberg(pvalue: np.typing.ArrayLike) -> np.ndarray:
    """
    Adjusts p-values with the Benjamini-Hochberg method.

    Parameters
    ----------
    pvalue : array_like
        A 1D array of p-values

    Returns
    -------
    All adjusted p-values
    """
    pvalue = np.asarray(pvalue)
    m = len(pvalue)
    # sort in ascending
    sorted_idx = np.argsort(pvalue)
    p_sorted = np.sort(pvalue)

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

    return adjusted


def benjamini_yekutieli(pvalue: np.typing.ArrayLike) -> np.ndarray:
    """
    Adjusts p-values with the Benjamini-Yekutieli method.

    Parameters
    ----------
    pvalue : array_like
        A 1D array of p-values

    Returns
    -------
    Adjusted p-values.

    Notes
    -----
    The formula to adjust each p-value is as follows:

    $p = min(1, min_{j >= i}(\frac {m * c(m) * p_{(j)}} {j}))$

    where *m* is the total number of tests, *p_(j)* is the jth smallest p-value, *c(m)* is the adjustment factor (this is determined by the formula sum_k=1_to_m(1 / k)) and *j* rank (or integer index) -- the position of the item in the sorted list.

    This adjustment method is a more conservative False Discovery Rate (FDR) correction than the Benjamini-Hochberg method, ensuring control even when test statistics are arbitrarily dependent.
    """
    pvalue = np.asarray(pvalue)
    m = len(pvalue)
    # sort
    sorted_idx = np.argsort(pvalue)
    pvalue_sorted = pvalue[sorted_idx]
    c = np.sum(1 / np.arange(1, m + 1))

    adjusted = (m * c * pvalue_sorted) / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1)

    result = np.empty_like(adjusted)
    result[sorted_idx] = adjusted
    return result


def hochberg(pvalue: np.typing.ArrayLike) -> np.ndarray:
    """
    Adjusts p-values with the Hochberg method.

    Parameters
    ----------
    pvalue : array_like
        A 1D array of p-values

    Returns
    -------
    Adjusted p-values

    Notes
    -----
    This is not to be confused with the *Benjamini-Hochberg* method (see `anovapy._p_adjust.benjamini`). This test controls the FWER (Family Wise Error Rate). The formula to adjust p-values is:

    $p = min((m - j + 1)p_{(j)})$

    where *m* are the total number of tests, *j* is the rank of the item, and *p_j* is the unadjusted p-value.

    References
    ----------
    .. [1] Hochberg, Y. (1988). A sharper bonferroni procedure for multiple tests of significance.
    """
    pvalue = np.asarray(pvalue)
    m = len(pvalue)
    sorted_idx = np.argsort(pvalue)
    pvalue_sorted = pvalue[sorted_idx]

    adjusted = (m - np.arange(m)) * pvalue_sorted
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1)

    result = np.empty_like(adjusted)
    result[sorted_idx] = adjusted
    return result


def adjust_pvalue(
    pvalue: np.typing.ArrayLike, adjust: PValueAdjustmentMethod
) -> np.ndarray:
    """
    Adjusts the p-value with the given adjustment method

    Parameters
    ----------
    pvalue : array_like
        The p-values to adjust
    adjust : PValueAdjustmentMethod
        The method to adjust p-values
        - 'bonferroni' : adjust p-values with the Bonferroni method
        - 'holm' : adjust p-values with the Holm-Bonferroni method
        - 'sidak' : adjust p-values with the Sidak method
        - 'bejamini' : adjust p-values with the Benjamini-Hochberg method
        - 'benjamini_yekutieli' : adjust p-values with the Benjamini-Yekutieli method
        - 'hochberg' : adjust p-values with the Hochberg method

    Returns
    -------
    Adjusted p-values

    Raises
    ------
    ValueError if `adjust` is not valid
    """
    if adjust == "none":
        return pvalue  # do nothing!
    valid_types = {
        "bonferroni": bonferroni,
        "holm": holm,
        "sidak": sidak,
        "benjamini": benjamini_hochberg,
        "benjamini_yekutieli": benjamini_yekutieli,
        "hochberg": hochberg,
    }

    if adjust not in valid_types:
        raise ValueError(f"No such p-value adjustment method '{adjust}' exists.")

    return valid_types[adjust](pvalue)
