# tests N-Way ANOVA

from itertools import combinations

import numpy as np

# lets test directly against statsmodels
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from anovapy.nway import art, f_nway, tukey_hsd

# some data being reused
# it is from: https://www.statology.org/three-way-anova-python/
height = np.array(
    [
        7,
        7,
        8,
        8,
        7,
        6,
        6,
        5,
        6,
        5,
        5,
        5,
        4,
        5,
        4,
        3,
        3,
        4,
        3,
        3,
        6,
        6,
        5,
        4,
        5,
        4,
        5,
        4,
        4,
        3,
        2,
        2,
        1,
        4,
        4,
        2,
        1,
        1,
        2,
        1,
    ]
)
program = np.repeat([1, 2], 20)
gender = np.tile(np.repeat(["M", "F"], 10), 2)
division = np.tile(np.repeat([1, 2], 5), 4)


def test_nway():

    rng = np.random.default_rng(42)
    for num_factors in range(3, 6):
        # unequal replication makes the Type I sums of squares order-sensitive
        cells = np.indices((2,) * num_factors).reshape(num_factors, -1).T
        replicates = rng.integers(2, 6, size=len(cells))
        observations = np.repeat(cells, replicates, axis=0)
        factors = [observations[:, i] for i in range(num_factors)]
        dependent = rng.normal(size=len(observations))

        data = {f"factor_{i}": factor for i, factor in enumerate(factors)}
        data["dependent"] = dependent
        terms = [
            ":".join(f"C(factor_{i})" for i in effect)
            for order in range(1, num_factors + 1)
            for effect in combinations(range(num_factors), order)
        ]
        formula = "dependent ~ " + " + ".join(terms)
        model = ols(formula, data=pd.DataFrame(data)).fit()
        statsmodels_result = sm.stats.anova_lm(model, typ=1)

        result = f_nway(*factors, dependent=dependent)
        np.testing.assert_allclose(
            result.statistic, statsmodels_result["F"].iloc[:-1], rtol=1e-8
        )
        np.testing.assert_allclose(
            result.pvalue, statsmodels_result["PR(>F)"].iloc[:-1], rtol=1e-8
        )


def test_art():
    # for art we have to manually test it
    # let's try three way art anova

    res = art(program, gender, division, dependent=height)

    expected_dfs = np.repeat([1], 7)
    np.testing.assert_allclose(res.df, expected_dfs, rtol=1e-2)
    np.testing.assert_allclose(res.df_error, 32, rtol=1e-2)

    # note: results were checked manually
    # also have some debugging, where ARTool rounds to the 8th place and AnovaPy doesn't, some values are expected to be slightly different
    # however when we also round tot he 8th place during computation in AnovaPy, we get the same result
    # therefore, to give high precision, AnovaPy doesn't round simply

    expected_stats = np.array(
        [75.09855, 95.80646, 43.44347, 0.16832, 0.69614, 0.15827, 0.27174]
    )
    expected_pvals = np.array(
        [6.6622e-10, 3.8189e-11, 1.9829e-07, 0.68434, 0.41027, 0.69339, 0.60576]
    )

    np.testing.assert_allclose(res.statistic, expected_stats, rtol=1e-2)
    np.testing.assert_allclose(res.pvalue, expected_pvals, rtol=1e-2)


def test_tukey():
    # this was all tested against R's TukeyHSd function
    # same data is reused as the test_art function
    res = tukey_hsd(program, gender, division, dependent=height)

    # TukeyHSD in R only gives p-value not stat
    # though if the p-value is correct, most likely the stat is correct too
    expected_pvals = np.array([2.984e-09])

    np.testing.assert_allclose(res.pvalue, expected_pvals, rtol=1e-5, atol=1e-10)

    res = tukey_hsd(program, gender, division, dependent=height, term=(1,))

    expected_pvals = np.array([1.824e-12])

    np.testing.assert_allclose(res.pvalue, expected_pvals, rtol=1e-5, atol=1e-10)

    res = tukey_hsd(program, gender, division, dependent=height, term=(2,))

    expected_pvals = np.array([1.2e-06])

    np.testing.assert_allclose(res.pvalue, expected_pvals, rtol=1e-4, atol=1e-12)

    res = tukey_hsd(program, gender, division, dependent=height, term=(0, 1))

    # expected_pvals = np.array([0.0000137, 0.0000000, 0.1714595, 0.0000000, 0.0000000, 0.0000137])
    expected_pvals = np.array(
        [
            0.000013724666194,
            0.000000035719319,
            0.171459503666604,
            0.000000000000145,
            0.000000035719319,
            0.000013724666194,
        ]
    )

    np.testing.assert_allclose(
        np.sort(res.pvalue), np.sort(expected_pvals), rtol=1e-5, atol=1e-10
    )

    res = tukey_hsd(program, gender, division, dependent=height, term=(0, 2))

    expected_pvals = np.array(
        [0.0000024, 0.0001863, 0.0000000, 0.4449346, 0.0053072, 0.0000783]
    )

    np.testing.assert_allclose(
        np.sort(res.pvalue), np.sort(expected_pvals), rtol=1e-5, atol=1e-10
    )

    res = tukey_hsd(program, gender, division, dependent=height, term=(1, 2))

    expected_pvals = np.array(
        [
            0.000000015820160,
            0.002360361206795,
            0.005307217009861,
            0.000000000001242,
            0.000439937107696,
            0.000000081508206,
        ]
    )

    np.testing.assert_allclose(
        np.sort(res.pvalue), np.sort(expected_pvals), rtol=1e-5, atol=1e-10
    )

    res = tukey_hsd(program, gender, division, dependent=height, term=(0, 1, 2))

    expected_pvals = np.array(
        [
            0.003680733885891,
            0.000030004972385,
            0.899866256804302,
            0.088565352450705,
            0.000002672021465,
            0.417272240944154,
            0.899866256804304,
            0.000000000348096,
            0.000101454694615,
            0.899866256804303,
            0.208433704707262,
            0.000008910056032,
            0.088565352450705,
            0.001136473365062,
            0.000000008442450,
            0.000000000001196,
            0.011424084598022,
            0.000000812746538,
            0.003680733885891,
            0.000000079401271,
            0.988215595664208,
            0.208433704707262,
            0.011424084598022,
            0.000341840597100,
            0.683723132886212,
            0.000000008442450,
            0.000101454694615,
            0.033274417299843,
        ]
    )

    np.testing.assert_allclose(
        np.sort(res.pvalue), np.sort(expected_pvals), rtol=1e-5, atol=1e-10
    )
