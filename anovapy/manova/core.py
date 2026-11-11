from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import linalg, stats

__all__ = ["f_manova", "transform_and_run_manova", "transform_dependent"]

ManovaTest = Literal["wilks", "pillai", "hotelling_lawley", "roy"]


@dataclass
class ManovaResult:
    """
    Results for `anovapy.manova.f_manova`.

    Attributes
    ----------
    statistic : float
        The selected multivariate statistic (note that this will be computed based on what the `test` parameter was set to in `f_manova`)
    pvalue : float
        Computed p-value from the F-distribution
    f_stat : float
        The F-Statistic
    df1 : float
        Degrees of Freedom (numerator)
    df2 : float
        Degrees of Freedom (denominator)
    """

    statistic: float
    pvalue: float
    f_stat: float
    df1: float
    df2: float

    def __iter__(self):
        """Allow `statistic, pvalue = f_manova(...)`."""
        yield self.statistic
        yield self.pvalue

    def __repr__(self):
        return (
            f"ManovaResult("
            f"  statistic={self.statistic}, "
            f"  pvalue={self.pvalue}, "
            f"  f_stat={self.f_stat}, "
            f"  df1={self.df1}, "
            f"  df2={self.df2}"
            f")"
        )


def _move_observation_axis(data: np.ndarray, axis: int) -> np.ndarray:
    # MANOVA calculations below consistently treat rows as observations and
    # columns as dependent variables, regardless of the caller's axis choice.
    if data.ndim not in (1, 2):
        raise ValueError("Data must be a 1D or 2D array.")
    if axis not in range(-data.ndim, data.ndim):
        raise ValueError(f"axis must be valid for a {data.ndim}D input.")
    # Moving the observation axis to the front leaves variables in columns so
    # a 1D response needs a singleton variable dimension for matrix operations
    moved = np.moveaxis(data, axis, 0)
    return moved[:, None] if moved.ndim == 1 else moved


def _missing_labels(labels: np.ndarray) -> np.ndarray:
    try:
        # np.isnan handles numeric labels (including floating-point NaN)
        missing = np.asarray(np.isnan(labels), dtype=bool)
        if missing.shape == labels.shape:
            return missing
    except TypeError:
        # Non-numeric label types do not support np.isnan; for those, None is
        # the missing-label sentinel
        pass
    return np.asarray([label is None for label in labels], dtype=bool)


def _f_approximation(
    statistic: float, roots: np.ndarray, p: int, q: int, v: int, test: ManovaTest
) -> tuple[float, float, float]:
    """
    Return the conventional approximate F statistic and its degrees of freedom.
    """
    # p is the number of dependent variables, q the hypothesis degrees of
    # freedom (groups - 1), and v the residual degrees of freedom. These
    # quantities parameterize the standard F approximations for each criterion
    s = min(p, q)
    m = (abs(p - q) - 1.0) / 2.0
    n = (v - p - 1.0) / 2.0

    if test == "wilks":
        # Wilks' lambda is the product of (1 - root). Rao's transformation
        # converts it to an approximate F with sample-size-adjusted degrees of
        # freedom. A nonpositive lambda maps to an infinite F statistic
        t_denominator = p * p + q * q - 5.0
        t = np.sqrt((p * p * q * q - 4.0) / t_denominator) if t_denominator > 0 else 1.0
        df1 = p * q
        df2 = (v - (p - q + 1.0) / 2.0) * t - (p * q - 2.0) / 2.0
        f_stat = (
            np.inf
            if statistic <= 0.0
            else ((statistic ** (-1.0 / t) - 1.0) * df2 / df1)
        )
    elif test == "pillai":
        # Pillai's trace sums the canonical roots. Its approximate F scaling
        # uses s (the maximum possible number of nonzero roots) and the shared
        # m/n terms to determine its degrees of freedom
        df1 = s * (2.0 * m + s + 1.0)
        df2 = s * (2.0 * n + s + 1.0)
        f_stat = (df2 / df1) * statistic / (s - statistic)
    elif test == "hotelling_lawley":
        # Hotelling-Lawley trace sums root/(1-root). For n > 0, the
        # finite-sample approximation uses b and c to derive its F degrees of
        # freedom and scaling; excluded boundary cases have no finite transform
        if n > 0.0:
            if n == 1.0:
                raise ValueError(
                    "The sample size does not permit a finite Hotelling-Lawley "
                    "F approximation."
                )
            b = ((p + 2.0 * n) * (q + 2.0 * n)) / (2.0 * (2.0 * n + 1.0) * (n - 1.0))
            df1 = p * q
            if b == 1.0:
                raise ValueError(
                    "The sample size does not permit a finite Hotelling-Lawley "
                    "F approximation."
                )
            df2 = 4.0 + (p * q + 2.0) / (b - 1.0)
            c = (df2 - 2.0) / (2.0 * n)
            if c == 0.0:
                raise ValueError(
                    "The sample size does not permit a finite Hotelling-Lawley "
                    "F approximation."
                )
            f_stat = (df2 / df1) * statistic / c
        else:
            # when the finite-sample parameter is nonpositive, use the
            # alternate approximation expressed in terms of s, m, and n
            df1 = s * (2.0 * m + s + 1.0)
            df2 = s * (s * n + 1.0)
            f_stat = (df2 / df1 / s) * statistic
    else:
        # Roy's criterion uses only the largest canonical root
        # so convert that root to root/(1-root), then scale it using Roy's approximate degrees of freedom
        greatest_root = np.max(roots / (1.0 - roots))
        df1 = max(p, q)
        df2 = v - max(p, q) + q
        f_stat = (df2 / df1) * greatest_root

    # a usable F distribution requires positive finite degrees of freedom so we
    # reject invalid transformed statistics rather than returning misleading
    # inferential results
    if not np.isfinite(df1) or not np.isfinite(df2) or df1 <= 0.0 or df2 <= 0.0:
        raise ValueError(
            "The sample size is too small for a positive F-approximation "
            "degrees of freedom."
        )
    if np.isnan(f_stat) or f_stat < 0.0:
        raise ValueError("The data do not produce a valid F approximation.")
    return f_stat, df1, df2


def f_manova(
    data: np.ndarray,
    groups: np.ndarray,
    *,
    test: ManovaTest = "wilks",
    axis: int = 0,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> ManovaResult:
    """
    Run a One-Way Multivariate Analysis of Variance (MANOVA).

    The recommended form is `f_manova(data, labels)`: `data` contains one
    observation per row and one dependent variable per column, while `labels`
    contains one categorical group label per observation. A 1D `data` array
    is treated as a single dependent variable. The number of groups is inferred
    from the distinct labels; each label corresponds to one independent sample.

    For compatibility, separate 2D sample matrices can also be passed as
    ``f_manova(group1, group2, ...)``. Each matrix has observations on ``axis``
    and dependent variables on the other axis. Separate-group inputs must all
    be 2D; use the data-and-labels form for 1D samples.

    Parameters
    ----------
    data : array_like
        A 2D array of shape (n_observations, n_variables) containing the
        dependent variables.
    groups : array_like
        A 1D array of shape (n_observations,) containing categorical group
        labels for each observation.
    test : {'wilks', 'pillai', 'hotelling_lawley', 'roy'}, optional
        The multivariate test statistic to compute. Options are:
        - 'wilks': Wilks' lambda (default)
        - 'pillai': Pillai's trace
        - 'hotelling_lawley': Hotelling-Lawley trace
        - 'roy': Roy's greatest root
    axis : int, optional
        The axis along which observations are oriented. Defaults to 0
        (rows as observations).
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle NaN values in observations.
        - 'propagate': returns NaN if NaNs are present.
        - 'omit': drops observations containing NaNs.
        - 'raise': raises a ValueError.

    Returns
    -------
    ManovaResult
        The selected multivariate statistic, its approximate F statistic,
        numerator and denominator degrees of freedom, and p-value.

    Raises
    ------
    ValueError
        If inputs are malformed, fewer than two groups remain, or the sample
        does not provide enough residual degrees of freedom for MANOVA.

    References
    ----------
    .. [1] Hair, J.F. (2011). Multivariate Data Analysis: An Overview. In: Lovric, M. (eds) International Encyclopedia of Statistical Science. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-04898-2_395
    .. [2] Rencher, Alvin C, and William F Christensen. Methods of Multivariate Analysis. 2012, https://doi.org/10.1002/9781118391686.
    .. [3] Zaiontz, Charles. "MANOVA Basic Concepts." Real-Statistics.Com, 2014, https://real-statistics.com/multivariate-statistics/multivariate-analysis-of-variance-manova/manova-basic-concepts/. Accessed 11 Nov. 2026.
    .. [4] IBM SPSS Statistics 24 Algorithms. https://public.dhe.ibm.com/software/analytics/spss/documentation/statistics/24.0/en/client/Manuals/IBM_SPSS_Statistics_Algorithms.pdf. Accessed 11 Nov. 2026.
    """
    # validate
    if test not in {"wilks", "pillai", "hotelling_lawley", "roy"}:
        raise ValueError(
            "test must be one of {'wilks', 'pillai', 'hotelling_lawley', 'roy'}"
        )
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")

    # normalize both forms to a list of (observations, variables) arrays so the
    # subsequent one-way MANOVA calculations can use a common path
    raw_inputs = [data]
    if groups is not None:
        raw_inputs.append(groups)
    # convert everything into a NumPy array as always
    arrays = [np.asarray(value) for value in raw_inputs]

    separate_groups = len(arrays) >= 2 and all(array.ndim == 2 for array in arrays)
    labels = None
    if separate_groups:
        # each supplied matrix is one independent group; normalize its
        # observation axis and dtype for the SSCP matrix calculations
        samples = [
            _move_observation_axis(np.asarray(array, dtype=np.float64), axis)
            for array in arrays
        ]
    else:
        if len(arrays) != 2 or arrays[1].ndim != 1:
            raise ValueError(
                "Pass pooled observations with a 1D group-label array, or pass "
                "at least two separate 2D group matrices."
            )

        # in pooled form, labels assign each observation to a group
        samples = [
            _move_observation_axis(np.asarray(arrays[0], dtype=np.float64), axis)
        ]
        labels = np.asarray(raw_inputs[1])
        if labels.ndim != 1:
            raise ValueError("groups must be a 1D label for each observation.")
        if labels.size != samples[0].shape[0]:
            raise ValueError(
                "groups must contain exactly one label for each observation."
            )
        # a missing group assignment cannot contribute to group means or to
        # the between-group hypothesis sum-of-squares matrix.
        missing_labels = _missing_labels(labels)
        if missing_labels.any():
            raise ValueError("Group labels must not contain missing values.")

    # validate shapes and infinities before computing means or cross-products
    # NaN handling is separate so its policy applies consistently to both
    # pooled and separate-group inputs
    if any(sample.shape[0] == 0 or sample.shape[1] == 0 for sample in samples):
        raise ValueError(
            "Each input must contain at least one observation and variable."
        )
    n_variables = samples[0].shape[1]
    if any(sample.shape[1] != n_variables for sample in samples[1:]):
        raise ValueError("All groups must have the same number of dependent variables.")
    if any(np.isinf(sample).any() for sample in samples):
        raise ValueError("Input observations must not contain infinite values.")

    has_nan = any(np.isnan(sample).any() for sample in samples)
    if has_nan and nan_policy == "raise":
        raise ValueError("The input contains NaN values.")
    if has_nan and nan_policy == "propagate":
        return ManovaResult(
            statistic=np.nan, pvalue=np.nan, f_stat=np.nan, df1=np.nan, df2=np.nan
        )
    if has_nan:
        # omit whole observations, not individual cells: all responses from
        # one observation must remain together in the multivariate model
        if labels is None:
            samples = [sample[~np.isnan(sample).any(axis=1)] for sample in samples]
        else:
            keep = ~np.isnan(samples[0]).any(axis=1)
            samples[0] = samples[0][keep]
            labels = labels[keep]

    if labels is not None:
        try:
            # convert comparable labels to integer group indices, then split
            # the pooled observations into the samples used by the formulas
            _, label_codes = np.unique(labels, return_inverse=True)
        except TypeError as exc:
            raise ValueError(
                "Group labels must be mutually comparable values."
            ) from exc
        samples = [
            samples[0][label_codes == code]
            for code in range(int(label_codes.max()) + 1)
        ]

    if len(samples) < 2 or any(sample.shape[0] == 0 for sample in samples):
        raise ValueError("MANOVA requires at least two non-empty groups.")

    # in a one-way design, p is the response dimension, q is the between-group
    # degrees of freedom, and v is the within-group residual degrees of freedom
    n_by_group = np.asarray([sample.shape[0] for sample in samples], dtype=np.float64)
    n_total = int(n_by_group.sum())
    n_groups = len(samples)
    p = n_variables
    q = n_groups - 1
    v = n_total - n_groups
    if v < p:
        raise ValueError(
            "MANOVA requires at least as many residual degrees of freedom "
            "as dependent variables."
        )

    # E (error) is the pooled within-group SSCP matrix:
    # sum_g sum_i (x_gi - mean_g)(x_gi - mean_g)^T.
    # H (hypothesis) is the between-group SSCP matrix:
    # sum_g n_g (mean_g - grand_mean)(mean_g - grand_mean)^T.
    # these are the multivariate counterparts of within-group and between-group
    # sums of squares in the one-way ANOVA decomposition.
    means = np.asarray([sample.mean(axis=0) for sample in samples])
    grand_mean = np.average(means, axis=0, weights=n_by_group)
    error = np.zeros((p, p), dtype=np.float64)
    hypothesis = np.zeros((p, p), dtype=np.float64)
    for sample, mean, n in zip(samples, means, n_by_group):
        centered = sample - mean
        error += centered.T @ centered
        mean_difference = mean - grand_mean
        hypothesis += n * np.outer(mean_difference, mean_difference)
    # symmetrize before calling routines for symmetric generalized eigenproblems;
    # this removes only floating-point roundoff asymmetry
    error = (error + error.T) / 2.0
    total_effect = error + hypothesis
    total_effect = (total_effect + total_effect.T) / 2.0

    # within-group SSCP matrix must be full rank to distinguish all p
    # dependent-variable directions and solve the MANOVA eigenvalue problem
    if np.linalg.matrix_rank(error) < p:
        raise ValueError(
            "The within-group error matrix is singular; MANOVA requires "
            "independent dependent variables and sufficient residual data."
        )
    try:
        # generalized eigenvalues of (H + E)^-1 H are the canonical roots:
        # each represents the share of total multivariate variation associated
        # with group differences
        roots = linalg.eigvalsh(hypothesis, total_effect)
    except linalg.LinAlgError as exc:
        raise ValueError("The MANOVA sums-of-squares matrix is singular.") from exc
    # roundoff can put a theoretically bounded root slightly outside [0, 1] so
    # keep roots in range and strictly below 1 because later criteria use 1-root
    roots = np.clip(np.real(roots), 0.0, np.nextafter(1.0, 0.0))

    # derive the selected criterion from the same roots: Wilks is the product
    # of (1 - root), Pillai is their sum, Hotelling-Lawley sums root/(1-root),
    # and Roy uses the largest root/(1-root). Each is then converted to an
    # approximate F statistic for hypothesis testing.
    if test == "wilks":
        statistic = np.prod(1.0 - roots)
    elif test == "pillai":
        statistic = np.sum(roots)
    elif test == "hotelling_lawley":
        statistic = np.sum(roots / (1.0 - roots))
    else:
        statistic = np.max(roots / (1.0 - roots))

    f_stat, df1, df2 = _f_approximation(statistic, roots, p, q, v, test)
    # finally derive p-value
    pvalue = stats.f.sf(f_stat, df1, df2)
    return ManovaResult(
        statistic=statistic, pvalue=pvalue, f_stat=f_stat, df1=df1, df2=df2
    )


def transform_dependent(
    dependent_vars: np.typing.ArrayLike,
    *,
    method: Literal["log", "sqrt", "boxcox"] = "log",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> np.typing.ArrayLike:
    """
    Transform the dependent variables for MANOVA analysis.

    This is typically used when assumptions of MANOVA are not met. The idea is to transform the dependent variables to better meet the assumptions of MANOVA, such as normality and homogeneity of variance.

    **Note that however you should always re-check the assumptions after transformation.**

    Parameters
    ----------
    dependent_vars : array_like
        A 2D array of shape (n_observations, n_variables) containing the dependent variables.
    method : {'log', 'sqrt', 'boxcox'}
        The method of transformation to apply. Defaults to 'log'
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle inputs with NaN values.

    Returns
    -------
    The transformed dependent variable

    References
    ----------
    .. [1] Box, G. E. P., and D. R. Cox. "An Analysis of Transformations." Journal of the Royal Statistical Society Series B: Statistical Methodology, vol. 26, no. 2, Oxford University Press (OUP), July 1964, pp. 211-43, https://doi.org/10.1111/j.2517-6161.1964.tb00553.x.
    .. [2] Jaeger TF. Categorical Data Analysis: Away from ANOVAs (transformation or not) and towards Logit Mixed Models. J Mem Lang. 2008 Nov;59(4):434-446. doi: 10.1016/j.jml.2007.11.007. PMID: 19884961; PMCID: PMC2613284.
    .. [3] Lee DK. Data transformation: a focus on the interpretation. Korean J Anesthesiol. 2020 Dec;73(6):503-508. doi: 10.4097/kja.20137. Epub 2020 Nov 20. PMID: 33271009; PMCID: PMC7714623.
    """
    # validate
    if nan_policy not in {"propagate", "omit", "raise"}:
        raise ValueError("nan_policy must be one of {'propagate', 'omit', 'raise'}.")
    # handle nan policy on all arrays
    # transform them as we go
    transformed_vars = []
    for var in dependent_vars:
        var = np.asarray(var, dtype=float)
        # check nan
        has_nan = np.isnan(var).any()
        if has_nan:
            if nan_policy == "omit":
                var = var[~np.isnan(var)]
            elif nan_policy == "raise" and np.isnan(var).any():
                raise ValueError("Input contains NaN values.")
            else:
                # propagate (return np.nan)
                transformed_vars.append(var)
                continue

        if method == "log":
            transformed_var = np.log(var)
        elif method == "sqrt":
            transformed_var = np.sqrt(var)
        elif method == "boxcox":
            # Box-Cox transformation requires positive data
            if np.any(var <= 0):
                raise ValueError("Box-Cox transformation requires positive data.")
            transformed_var, _ = stats.boxcox(var)
        else:
            raise ValueError(f"Unknown transformation method: {method}")

        transformed_vars.append(transformed_var)
    return np.array(transformed_vars)


def transform_and_run_manova(
    dependent: np.typing.ArrayLike,
    independent: np.typing.ArrayLike,
    *,
    test: ManovaTest = "wilks",
    method: Literal["log", "sqrt", "boxcox"] = "log",
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> ManovaResult:
    """
    Transforms the dependent variables and automatically runs the One Way MANOVA (see `anovapy.manova.f_manova`).

    Parameters
    ----------
    dependent : array_like
        A 2D array of shape (n_observations, n_variables) containing the
        dependent variables.
    independent : array_like
        A 1D array of shape (n_observations,) containing categorical group
        labels for each observation.
    test : {'wilks', 'pillai', 'hotelling_lawley', 'roy'}, optional
        The multivariate test statistic to compute. Options are:
        - 'wilks': Wilks' lambda (default)
        - 'pillai': Pillai's trace
        - 'hotelling_lawley': Hotelling-Lawley trace
        - 'roy': Roy's greatest root
    method : {'log', 'sqrt', 'boxcox'}
        The method to use for transforming the dependent variables.
    nan_policy : {'propagate', 'omit', 'raise'}
        Defines how to handle inputs with NaN values.

    Returns
    -------
    ManovaResult
        A dataclass containing the results.

    Notes
    -----
    This function is the equivalent of calling `transform_dependent` and then `f_manova` immediately.
    """
    # transform first
    transformed = transform_dependent(dependent, method=method, nan_policy=nan_policy)

    # pass it on to the main function
    return f_manova(transformed, independent, test=test, nan_policy=nan_policy)
