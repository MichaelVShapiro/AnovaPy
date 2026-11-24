# tests N-Way ANOVA

from itertools import combinations

import numpy as np

# lets test directly against statsmodels
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

from anovapy.nway import art, f_nway


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
    # we will utilize the same data here: https://www.statology.org/three-way-anova-python/
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
