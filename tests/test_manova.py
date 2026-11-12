from typing import cast

import numpy as np
import pytest
from scipy.stats import f_oneway
from statsmodels.multivariate.manova import MANOVA

from anovapy.manova import f_manova
from anovapy.manova.core import ManovaTest


def _statsmodels_result(data, labels, test):
    """Use statsmodels as a test oracle for the selected group effect."""
    labels = np.asarray(labels)
    _, codes = np.unique(labels, return_inverse=True)
    n_groups = int(codes.max()) + 1
    design = np.column_stack(
        (
            np.ones(labels.size),
            (codes[:, None] == np.arange(1, n_groups)).astype(float),
        )
    )
    contrast = np.column_stack((np.zeros(n_groups - 1), np.eye(n_groups - 1)))
    result = MANOVA(endog=np.asarray(data), exog=design).mv_test(
        [("group effect", contrast)]
    )
    names = {
        "wilks": "Wilks' lambda",
        "pillai": "Pillai's trace",
        "hotelling_lawley": "Hotelling-Lawley trace",
        "roy": "Roy's greatest root",
    }
    return result.results["group effect"]["stat"].loc[names[test]]


@pytest.mark.parametrize(
    "test",
    ["wilks", "pillai", "hotelling_lawley", "roy"],
)
@pytest.mark.parametrize("seed", [2, 17, 91])
def test_matches_statsmodels_for_random_multivariate_data(test, seed):
    rng = np.random.default_rng(seed)
    labels = np.repeat(["control", "low", "high"], [18, 21, 24])
    offsets = np.array([[0.0, 0.0, 0.0], [0.4, -0.2, 0.5], [0.9, 0.3, -0.4]])
    data = (
        rng.normal(size=(labels.size, 3))
        + offsets[np.unique(labels, return_inverse=True)[1]]
    )

    actual = f_manova(data, labels, test=test)
    expected = _statsmodels_result(data, labels, test)

    np.testing.assert_allclose(actual.statistic, expected["Value"], rtol=1e-9)
    np.testing.assert_allclose(actual.f_stat, expected["F Value"], rtol=1e-9)
    np.testing.assert_allclose(actual.pvalue, expected["Pr > F"], rtol=1e-9)
    np.testing.assert_allclose(actual.df1, expected["Num DF"], rtol=1e-9)
    np.testing.assert_allclose(actual.df2, expected["Den DF"], rtol=1e-9)


def test_accepts_one_dimensional_data_and_infers_groups_from_labels():
    rng = np.random.default_rng(4)
    labels = np.repeat(["a", "b", "c"], 20)
    data = rng.normal(size=labels.size) + np.repeat([0.0, 0.3, 0.8], 20)

    result_1d = f_manova(data, labels)
    result_2d = f_manova(data[:, None], labels)
    samples = [data[labels == label] for label in np.unique(labels)]
    expected = f_oneway(*samples)
    total = np.sum((data - data.mean()) ** 2)
    within = sum(np.sum((sample - sample.mean()) ** 2) for sample in samples)

    np.testing.assert_allclose(result_1d.statistic, within / total, rtol=1e-9)
    np.testing.assert_allclose(result_1d.f_stat, expected.statistic, rtol=1e-9)
    np.testing.assert_allclose(result_1d.pvalue, expected.pvalue, rtol=1e-9)
    assert result_1d == result_2d


def test_axis_selects_observation_dimension():
    rng = np.random.default_rng(52)
    labels = np.repeat([0, 1, 2], 15)
    data = rng.normal(size=(labels.size, 2)) + np.repeat(
        [[0.0, 0.0], [0.5, -0.4], [1.0, 0.2]], 15, axis=0
    )

    rows_are_observations = f_manova(data, labels, axis=0)
    columns_are_observations = f_manova(data.T, labels, axis=1)

    np.testing.assert_allclose(
        [
            rows_are_observations.statistic,
            rows_are_observations.f_stat,
            rows_are_observations.pvalue,
        ],
        [
            columns_are_observations.statistic,
            columns_are_observations.f_stat,
            columns_are_observations.pvalue,
        ],
    )


def test_nan_policy_propagate_omit_and_raise():
    data = np.array([[1.0], [1.5], [2.0], [4.0], [4.5], [np.nan], [5.5], [6.0]])
    labels = np.array(["a", "a", "a", "b", "b", "b", "b", "b"])

    propagated = f_manova(data, labels, nan_policy="propagate")
    omitted = f_manova(data, labels, nan_policy="omit")
    manually_omitted = f_manova(
        data[~np.isnan(data[:, 0])], labels[~np.isnan(data[:, 0])]
    )

    assert np.isnan(propagated.statistic)
    assert np.isnan(propagated.f_stat)
    assert np.isnan(propagated.pvalue)
    assert omitted == manually_omitted
    with pytest.raises(ValueError):
        f_manova(data, labels, nan_policy="raise")


@pytest.mark.parametrize(
    ("data", "labels"),
    [
        (np.ones((5, 2, 1)), np.zeros(5)),
        (np.ones((5, 2)), np.zeros(4)),
        (np.ones((5, 2)), np.array(["a", "a", "a", "a", "a"])),
    ],
)
def test_rejects_invalid_inputs(data, labels):
    with pytest.raises(ValueError):
        f_manova(data, labels)


def test_rejects_unknown_test_and_singular_error_matrix():
    rng = np.random.default_rng(8)
    labels = np.repeat([0, 1], 8)
    data = rng.normal(size=(labels.size, 2))
    data[:, 1] = 2 * data[:, 0]

    with pytest.raises(ValueError, match="test must be"):
        f_manova(data, labels, test=cast(ManovaTest, "invalid"))
    with pytest.raises(ValueError, match="singular"):
        f_manova(data, labels)


def test_rejects_sample_size_without_hotelling_lawley_approximation():
    rng = np.random.default_rng(12)
    data = rng.normal(size=(7, 2))
    labels = np.array(["a", "a", "a", "b", "b", "b", "b"])

    with pytest.raises(ValueError, match="Hotelling-Lawley"):
        f_manova(data, labels, test="hotelling_lawley")
