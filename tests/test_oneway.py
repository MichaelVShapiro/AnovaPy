# note that some functions here are not tested since they just make calls to scipy

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.formula.api import ols

from anovapy.oneway import (
    BrownResult,
    RepeatedMeasuresAnovaResult,
    brown,
    dunns,
    f_ancova,
    f_repeated,
    pairwise_ttest,
)

# this one is taken from: https://www.statology.org/repeated-measures-anova-by-hand/
tutorial_samples: list[np.ndarray] = [
    np.array([30, 14, 24, 38, 26]),  # Drug 1
    np.array([28, 18, 20, 34, 28]),  # Drug 2
    np.array([16, 10, 18, 20, 14]),  # Drug 3
]


def test_dunns_simple():
    # just some random numbers used
    group_a = np.array([1, 3, 4, 2, 3, 1, 4, 5, 6, 1, 2, 3])
    group_b = np.array([1, 4, 5, 6, 3, 3, 4, 2, 2, 2, 4, 4])
    group_c = np.array([1, 6, 7, 5, 6, 8, 4, 6, 9, 9, 1, 1])

    res = dunns(group_a, group_b, group_c)

    # we don't really care about the unadjusted p-value, so test the adjusted only for simplicity
    np.testing.assert_allclose(
        res.pvalue, np.array([1.0, 0.084469, 0.311385]), rtol=1e-2
    )

    # test holm
    res = dunns(group_a, group_b, group_c, adjust="holm")
    np.testing.assert_allclose(
        res.pvalue, np.array([0.5698, 0.0845, 0.2076]), rtol=1e-2
    )

    # test sidak
    res = dunns(group_a, group_b, group_c, adjust="sidak")
    np.testing.assert_allclose(
        res.pvalue, np.array([0.9204, 0.0821, 0.2802]), rtol=1e-2
    )

    # test benjamini adjustment method
    res = dunns(group_a, group_b, group_c, adjust="benjamini")
    np.testing.assert_allclose(
        res.pvalue, np.array([0.5698, 0.0845, 0.1557]), rtol=1e-2
    )


@pytest.mark.parametrize(
    ("samples", "correction", "expected"),
    [
        (
            tutorial_samples,
            None,
            RepeatedMeasuresAnovaResult(statistic=15.006, pvalue=0.00196),
        ),
        (
            tutorial_samples,
            "greenhouse",
            RepeatedMeasuresAnovaResult(statistic=15.006, pvalue=0.0045),
        ),
        (
            tutorial_samples,
            "huynh",
            RepeatedMeasuresAnovaResult(statistic=15.006, pvalue=0.001962),
        ),
        (
            tutorial_samples,
            "lower",
            RepeatedMeasuresAnovaResult(statistic=15.006, pvalue=0.017937),
        ),
    ],
)
def test_f_repeated(
    samples: list[np.ndarray], correction: str, expected: RepeatedMeasuresAnovaResult
):
    # get result
    res = f_repeated(*samples, correction=correction)

    np.testing.assert_allclose(res.statistic, expected.statistic, rtol=1e-2)
    np.testing.assert_allclose(res.pvalue, expected.pvalue, rtol=1e-2)


def test_adjusted_ttest():
    res = pairwise_ttest(*tutorial_samples)

    # to keep this test case short, test only the p-value
    np.testing.assert_allclose(
        res.pvalue, np.array([1.0, 0.041379, 0.035478]), rtol=1e-2
    )


@pytest.mark.parametrize(
    ("samples", "expected"),
    [(tutorial_samples, BrownResult(statistic=4.0417, pvalue=0.0555))],
)
def test_brown(samples: list[np.ndarray], expected: BrownResult):
    # get result
    # note that this function was tested against R's `onewaytests` package and all results were precomputed
    res = brown(*samples)

    np.testing.assert_allclose(res.statistic, expected.statistic, rtol=1e-2)
    np.testing.assert_allclose(res.pvalue, expected.pvalue, rtol=1e-2)


def test_ancova_tutorial():
    # tests the ancova function based on values given in: https://www.statology.org/ancova-excel/
    study_technique = np.repeat(["A", "B", "C"], 5)
    current_grade = np.array(
        [67, 88, 75, 77, 85, 92, 69, 77, 74, 88, 96, 91, 88, 82, 80]
    )  # covariate
    exam_score = np.array(
        [77, 89, 72, 74, 69, 78, 88, 93, 94, 90, 85, 81, 83, 88, 79]
    )  # dependent

    res = f_ancova(study_technique, exam_score, current_grade)

    np.testing.assert_allclose(res.statistic, 4.810, rtol=1e-2)
    np.testing.assert_allclose(np.round(res.pvalue, 3), 0.032, rtol=1e-3)

    np.testing.assert_allclose(res.covariate_stats, np.array([0.103]), rtol=1e-2)
    np.testing.assert_allclose(res.covariate_pvalues, np.array([0.754]), rtol=1e-2)


def test_ancova_covariate2():
    # tests with 2 covariates passed in
    # we will reuse the same data as test_ancova_tutorial
    # we will just add one more covariate that's all
    study_technique = np.repeat(["A", "B", "C"], 5)
    current_grade = np.array(
        [67, 88, 75, 77, 85, 92, 69, 77, 74, 88, 96, 91, 88, 82, 80]
    )  # covariate
    exam_score = np.array(
        [77, 89, 72, 74, 69, 78, 88, 93, 94, 90, 85, 81, 83, 88, 79]
    )  # dependent
    covariate2 = np.array([67, 88, 68, 74, 55, 45, 95, 100, 87, 89, 84, 80, 82, 78, 70])

    res = f_ancova(study_technique, exam_score, current_grade, covariate2)

    np.testing.assert_allclose(res.statistic, 4.358, rtol=1e-2)
    np.testing.assert_allclose(np.round(res.pvalue, 3), 0.044, rtol=1e-2)

    np.testing.assert_allclose(
        res.covariate_stats, np.array([0.416, 19.303]), rtol=1e-2
    )
    np.testing.assert_allclose(
        np.round(res.covariate_pvalues, 3), np.array([0.533, 0.001]), rtol=1e-2
    )


def test_against_statsmodels():
    # tests against statsmodels
    possible_labels = np.array(["A", "B", "C", "D", "E", "F", "G"])
    np.random.shuffle(possible_labels)
    for test in range(100):
        print(f"Test #{test + 1}")
        num_covariates = np.random.randint(1, 4)
        sample_size = np.random.randint(8, 100)
        ind = [np.random.choice(possible_labels) for _ in range(sample_size)]
        ind = np.array(ind)
        np.random.shuffle(ind)
        dependent = np.random.normal(loc=num_covariates * 4, scale=50, size=sample_size)
        covariates = [
            np.random.normal(loc=num_covariates * 4, scale=50, size=sample_size)
            for _ in range(num_covariates)
        ]

        eq = "dependent ~ C(independent)"

        d = {"dependent": dependent, "independent": ind}
        for i in range(len(covariates)):
            d[f"covariate_{i}"] = covariates[i]
            eq += f" + covariate_{i}"

        df = pd.DataFrame(d)

        try:
            res_anovapy = f_ancova(ind, dependent, *covariates)
        except ValueError:
            # then statsmodels should break too
            with pytest.raises(ValueError):
                model = ols(eq, data=df).fit()
                ancova_table = sm.stats.anova_lm(model, typ=2)

        # statsmodels res
        model = ols(eq, data=df).fit()
        ancova_table = sm.stats.anova_lm(model, typ=2)

        # extract all stats
        main_stat = ancova_table.iloc[0]["F"]
        main_pval = ancova_table.iloc[0]["PR(>F)"]

        covar_stats = [ancova_table.iloc[i]["F"] for i in range(1, num_covariates + 1)]
        covar_pvals = [
            ancova_table.iloc[i]["PR(>F)"] for i in range(1, num_covariates + 1)
        ]

        covar_stats = np.array(covar_stats)
        covar_pvals = np.array(covar_pvals)

        np.testing.assert_allclose(res_anovapy.statistic, main_stat, rtol=1e-2)
        np.testing.assert_allclose(res_anovapy.pvalue, main_pval, rtol=1e-2)

        np.testing.assert_allclose(res_anovapy.covariate_stats, covar_stats, rtol=1e-2)
        np.testing.assert_allclose(
            res_anovapy.covariate_pvalues, covar_pvals, rtol=1e-2
        )
