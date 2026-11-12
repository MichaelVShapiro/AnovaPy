# note that some functions here are not tested since they just make calls to scipy

import numpy as np
import pytest

from anovapy.oneway import (
    BrownResult,
    RepeatedMeasuresAnovaResult,
    brown,
    dunns,
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
