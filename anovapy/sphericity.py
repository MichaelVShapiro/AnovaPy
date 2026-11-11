# for checking if assumption is sphericity is met

from collections import namedtuple
from typing import Literal

import numpy as np
from scipy.linalg import helmert
from scipy.stats import chi2

__all__ = ["john", "mauchly"]

MauchlyResult = namedtuple("MauchlyResult", ["statistic", "pvalue"])
JohnResult = namedtuple("JohnResult", ["statistic", "pvalue"])


def mauchly(
    *samples: np.typing.ArrayLike,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> MauchlyResult:
    """
    Performs the Mauchly Test to check for the sphericity assumption.

    Parameters
    ----------
    *samples : array_like
        Samples measurements for each group.
    nan_policy : {'propagate', 'omit', 'raise'}
        How to handle NaN inputs. Defaults to 'propagate'

    Returns
    -------
    statistic : float
        The computed W statistic of the test
    pvalue : float
        The associated p-value from the chi-distribution

    Notes
    -----
    Values of W close to 1 indicate perfect sphericity, while values near 0 indicate severe violation. The W statistic is computed based on the formula:

    $W = det(S) / (trace(S) / m)^m$

    where *S* is the covariance matrix, *det* is the determinant of S, *trace(S)* is the sum of the variances of the measures, and *m* is the number of repeated measures.

    See Also
    --------
    john : John-Nagao-Sugiura test

    References
    ----------
    .. [1] Mauchly, John W. "Significance Test for Sphericity of a Normal N-Variate Distribution." The Annals of Mathematical Statistics, vol. 11, no. 2, June 1940, pp. 204-09, https://doi.org/10.1214/aoms/1177731915.
    .. [2] Lee, Sarah. "Ultimate Guide to Sphericity in MANOVA." Number Analytics, 14 May 2025, https://www.numberanalytics.com/blog/sphericity-mauchly-test-guide. Accessed 11 Nov. 2026.
    .. [3] Bobbitt, Zach. "Mauchly's Test of Sphericity: Definition & Example." Statology, 24 Aug. 2021, https://www.statology.org/mauchlys-test-of-sphericity/. Accessed 11 Nov. 2026.

    Examples
    --------
    >>> from anovapy.sphericity import mauchly
    >>> import numpy as np
    >>> sample_a = np.random.normal(size=60)
    >>> sample_b = np.random.normal(size=60)
    >>> sample_c = np.random.normal(size=60)
    >>> res = mauchly(sample_a, sample_b, sample_c)
    >>> print(f"W-Statistic: {res.statistic}, p-value: {res.pvalue}")
    """
    # convert everything into NumPy arrays as needed
    data = [np.asarray(i) for i in samples]

    # Mauchly's Test requires there to be at least three repeated measures
    if len(data) < 3:
        raise ValueError(
            "Mauchly's test requires at least 3 repeated measures conditions."
        )

    x = np.column_stack(data)  # shape: (n_subects, k_conditions)

    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # handle nan
    has_nan = np.isnan(x).any()
    if has_nan:
        if nan_policy == "propagate":
            return MauchlyResult(statistic=np.nan, pvalue=np.nan)
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        # omit
        x = x[~np.isnan(x).any(axis=1)]

    n, p = x.shape

    # compute sample covariance
    S = np.cov(x, rowvar=False)

    # build orthonormal contrast matrix M ((k-1) x k)
    M = helmert(p)

    # project covariance matrix into contrast space C ((k-1) x (k-1))
    C = M @ S @ M.T

    # now compute the W statistic
    d = p - 1
    det_C = np.linalg.det(C)
    trace_C = np.trace(C)

    if trace_C == 0 or det_C <= 0:
        W = 0.0
    else:
        W = det_C / ((trace_C / d) ** d)

    # bound W numerically to [0, 1]
    W = np.clip(W, 0.0, 1.0)

    # now calculate p-value
    # compute degrees of freedom
    df = int((d * (d + 1)) / 2 - 1)

    if W == 0 or df <= 0:
        return MauchlyResult(statistic=W, pvalue=0.0)

    f = 1.0 - (2 * d**2 + d + 2) / (6.0 * d * (n - 1))
    chi2_stat = -(n - 1) * f * np.log(W)
    p_value = chi2.sf(chi2_stat, df)

    return MauchlyResult(statistic=W, pvalue=p_value)


def john(
    *samples: np.typing.ArrayLike,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> JohnResult:
    """
    Does the John-Nagao-Sugiura test to check for sphericity in data.

    Parameters
    ----------
    *samples : array_like
        Samples measurements for each group.
    nan_policy : {'propagate', 'omit', 'raise'}
        How to handle NaN inputs. Defaults to 'propagate'

    Returns
    -------
    statistic : float
        The computed T statistic of the test
    pvalue : float
        The associated p-value from the chi-distribution

    Notes
    -----
    This function strictly returns chi squared test statistic T (in place of `statistic`), which is also used to help compute the p-value. Note that this differs from the V-statistic. The V-statistic, which is defined to be:

    $tr(S^2) / (tr(S))^2$

    where *S*  is the covariance matrix, and *trace(S/S^2)* is the sum of the variances of the measures.

    The V-statistic here more so serves as a descriptive statistic/effect size. It says how far the data are from sphericity. The T-statistic (what is return by this function), is defined to be:

    $T = (n / 2) * p^2 * (V - 1 / p)$

    where *n* refers to the number of subjects, and *p* refers to the number of repeated measures. The T-statistic here is the "actual" statistic of the test.

    References
    ----------
    .. [1] JOHN, S. "The Distribution of a Statistic Used for Testing Sphericity of Normal Distributions." Biometrika, vol. 59, no. 1, Apr. 1972, pp. 169-73, https://doi.org/10.1093/biomet/59.1.169.
    .. [2] Nagao, H. Properties of some test criteria for covariance matrix. Ann Inst Stat Math 28, 403-409 (1976). https://doi.org/10.1007/BF02504758
    .. [3] Sugiura, N. (1972). Locally Best Invariant Test for Sphericity and the Limiting Distributions. The Annals of Mathematical Statistics, 43(4), 1312-1316. http://www.jstor.org/stable/2239959
    .. [4] "More About Sphericity | Real Statistics Using Excel." Real-Statistics.Com, 28 Dec. 2012, https://real-statistics.com/anova-repeated-measures/sphericity/additional-information-sphericity/. Accessed 11 Nov. 2026.
    """
    # convert everything into NumPy arrays as needed
    data = [np.asarray(i) for i in samples]

    # validate
    if len(data) < 3:
        raise ValueError(
            "John-Nagao-Sugiura test requires at least 3 repeated measures conditions."
        )

    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    x = np.column_stack(data)  # shape: (n_subects, k_conditions)

    # handle nan
    has_nan = np.isnan(x).any()
    if has_nan:
        if nan_policy == "propagate":
            return JohnResult(statistic=np.nan, pvalue=np.nan)
        elif nan_policy == "raise":
            raise ValueError("Input contains NaN values.")
        # omit
        x = x[~np.isnan(x).any(axis=1)]

    n, p = x.shape

    # compute sample covariance matrix
    S = np.cov(x, rowvar=False)

    # build orthonormal contrast matrix M ((k-1) x k)
    M = helmert(p)

    # project covariance matrix into contrast space C ((k-1) x (k-1))
    C = M @ S @ M.T

    # apply to formula
    tr2 = np.linalg.trace(C @ C)
    tr = np.linalg.trace(C)

    # by the way, the V-statistic here is equal to:
    # V = tr2 / tr**2

    T = (n / 2) * ((p - 1) ** 2) * ((tr2 / tr**2) - (1 / (p - 1)))

    # compute degrees of freedom
    df = ((p - 1) * p) / 2 - 1

    # compute p-value
    # the T-statistic is basically already on the chi-squared scale
    # so no need to do anything too special just plug it in
    p_value = chi2.sf(T, df)

    return JohnResult(statistic=T, pvalue=p_value)
