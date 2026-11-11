# for testing Mardia's Test for normality

# we will also test other normality here

import numpy as np
import pytest

from anovapy.normal import (
    DoornikResult,
    EnergyResult,
    HenzeResult,
    MardiaResult,
    RoystonResult,
    ShapiroFranciaResult,
    doornik,
    energy,
    henze,
    mardia,
    royston,
    shapiro_francia,
)

# note: all expected values in these test cases were derived from R
# more specifically, functions were tested against the MVN package and mvnTest

sample1 = np.array(
    [[1.5, 2.3], [2.1, 2.9], [1.8, 2.1], [3.2, 4.0], [2.5, 3.1], [2.9, 3.8]]
)
sample2 = np.array(
    [
        [12, 13],
        [14, 15],
        [10, 9],
        [18, 14],
        [33, 12],
        [101, 67],
        [2, 3],
        [1, 2],
        [0, 0],
        [14, 12],
    ]
)
sample3 = np.array(
    [
        [1.5, 2.3, 2.3, 1.5],
        [9.8, 5.5, 2.2, 3.3],
        [4.4, 3.2, 0.1, -0.3],
        [4.4, 5.5, 6.6, 7.7],
        [6.4, 2.1, 1.1, 3.1],
        [0.4, 0.05, 18.3, 0.9],
    ]
)
sample4 = np.array(
    [
        [1.0, 2.0],
        [1.5, 2.3],
        [2.0, 2.8],
        [2.5, 3.1],
        [3.0, 3.5],
        [3.5, 4.2],
        [4.0, 4.5],
        [4.5, 5.0],
    ]
)

# since royston requires sample size to be at least 7 in R's MVN package
sample1_royston = np.array(
    [
        [1.5, 2.3],
        [2.1, 2.9],
        [1.8, 2.1],
        [3.2, 4.0],
        [2.5, 3.1],
        [2.9, 3.8],
        [3.1, 3.2],
        [1.1, 4.1],
    ]
)
sample3_royston = np.array(
    [
        [1.5, 2.3, 2.3, 1.5],
        [9.8, 5.5, 2.2, 3.3],
        [4.4, 3.2, 0.1, -0.3],
        [4.4, 5.5, 6.6, 7.7],
        [6.4, 2.1, 1.1, 3.1],
        [0.4, 0.05, 18.3, 0.9],
        [14.2, 10.9, 7.7, 9.9],
        [4.4, 5.5, 6.6, 5.4],
    ]
)

# generated via np.random.normal(size=60)
# this will be used when testing doornik-hansen test
sample_univariate = np.array(
    [
        0.72451721,
        -1.70481394,
        -0.72903186,
        1.50612899,
        1.32009541,
        0.18965231,
        -0.03695955,
        0.05499099,
        -0.52858561,
        -0.99783005,
        1.59820241,
        -0.9973,
        0.57723654,
        -0.02832004,
        -1.23571112,
        -1.26887188,
        0.92762482,
        -0.21628608,
        -0.51075382,
        1.72871171,
        -0.45874515,
        -0.27395179,
        0.90437154,
        -0.42858614,
        0.58000977,
        -0.48719451,
        -1.41598396,
        -0.5899318,
        -0.36092057,
        -1.59433453,
        -1.6815911,
        0.42837378,
        -0.64606894,
        -0.52872106,
        -0.36745123,
        -2.88107678,
        -0.64503301,
        0.54753659,
        1.72775804,
        -0.95252547,
        -2.45512448,
        1.00940998,
        1.19900397,
        0.44581831,
        -0.29756893,
        -0.75073424,
        -0.0718046,
        -0.18421839,
        -1.49643543,
        -0.28669743,
        -0.23313308,
        -0.97748807,
        -0.25604253,
        0.73160847,
        1.17467104,
        -0.83790142,
        -0.77761733,
        -0.36990854,
        -0.15339178,
        0.30537278,
    ]
)

# generated with np.random.normal(size=70)
sample_univariate2 = np.array(
    [
        0.58123281,
        1.12271915,
        0.31296543,
        -0.10038544,
        0.4369131,
        0.97538393,
        -1.58475807,
        -0.08634549,
        1.50732205,
        -0.89500984,
        0.54587895,
        0.18977902,
        -0.08834186,
        -0.253771,
        0.23120246,
        -0.0923818,
        -0.22620599,
        -0.21890866,
        -0.35673583,
        -0.06449225,
        0.44011737,
        0.38885666,
        -0.66599127,
        -0.03913377,
        0.76036773,
        -0.26218579,
        0.46702008,
        1.99217926,
        -0.42746972,
        1.97726203,
        -0.24772909,
        -2.38157562,
        0.81191171,
        -0.88005655,
        -0.15813154,
        -0.44099044,
        -0.34187378,
        0.79461401,
        0.51152998,
        0.10807838,
        0.72466517,
        -0.2931048,
        -0.8346172,
        1.57275735,
        1.61978506,
        0.36921177,
        -0.03478818,
        2.40462148,
        1.79415648,
        -0.06197367,
        0.04760701,
        1.75146388,
        -0.48700565,
        1.62426152,
        0.25353363,
        1.08279977,
        1.18835959,
        -0.25660068,
        -0.09731652,
        0.41544507,
        0.06730386,
        0.25760405,
        0.09735947,
        -1.38610245,
        -0.36690695,
        -0.17030564,
        -0.84638643,
        1.21971319,
        0.31000555,
        0.50930681,
    ]
)

# generated with np.random.exponential(size=60)
sample_univariate3 = np.array(
    [
        2.37051528,
        0.53503394,
        0.45978907,
        1.90478055,
        0.65489634,
        1.36444153,
        0.18824883,
        0.32506255,
        0.07770542,
        1.91232175,
        0.91107414,
        1.75354144,
        0.10170498,
        0.78972753,
        0.21736473,
        3.85214249,
        1.16490672,
        0.77118669,
        0.49074188,
        2.09806811,
        0.26144973,
        0.26693803,
        0.89876266,
        0.49739343,
        0.29654611,
        0.98139267,
        1.05875695,
        1.37333538,
        1.30610912,
        0.15639336,
        0.35641563,
        1.10238068,
        0.14555143,
        0.51396798,
        0.04196405,
        0.62615801,
        2.14083341,
        0.53557017,
        0.56327009,
        1.98243875,
        0.4549323,
        0.7190697,
        0.66475401,
        0.56374099,
        2.93546988,
        0.70599481,
        1.23678589,
        1.20271995,
        0.46238428,
        1.39594613,
        0.21917474,
        1.5421595,
        0.58549106,
        0.92523737,
        0.20369859,
        0.02761458,
        1.00654304,
        0.27842448,
        0.45734315,
        0.73956271,
    ]
)


@pytest.mark.parametrize(
    ("X", "expected"),
    [
        (sample1, MardiaResult(1.25, 1.25, 0.87, 3.95, -1.24, 0.21)),
        (sample2, MardiaResult(6.06, 10.1, 0.039, 10.29, 0.91, 0.36)),
        (sample3, MardiaResult(6.2, 6.2, 1, 11.75, -2.17, 0.03)),
        (sample4, MardiaResult(0.52, 0.69, 0.95, 3.74, -1.51, 0.13)),
    ],
)
def test_mardia(X: np.ndarray, expected: MardiaResult):
    # call to mardia
    res = mardia(X)

    # check all
    np.testing.assert_allclose(res.skew, expected.skew, rtol=1e-2)
    np.testing.assert_allclose(res.skew_statistic, expected.skew_statistic, rtol=1e-2)
    np.testing.assert_allclose(res.kurtosis, expected.kurtosis, rtol=1e-2)
    np.testing.assert_allclose(
        res.kurtosis_statistic, expected.kurtosis_statistic, rtol=1e-2
    )

    np.testing.assert_allclose(res.skew_pvalue, expected.skew_pvalue, rtol=3e-2)
    np.testing.assert_allclose(res.kurtosis_pvalue, expected.kurtosis_pvalue, rtol=3e-2)


@pytest.mark.parametrize(
    ("X", "expected"),
    [
        (sample1_royston, RoystonResult(0.671, 0.717)),
        (sample2, RoystonResult(18.057, 0.001)),
        (sample3_royston, RoystonResult(5.591, 0.182)),
        (sample4, RoystonResult(0.031, 0.871)),
    ],
)
def test_royston(X: np.ndarray, expected: RoystonResult):
    # call to royston
    res = royston(X)

    np.testing.assert_allclose(res.statistic, expected.statistic, rtol=3e-2)

    if (
        expected.pvalue == 0.001
    ):  # just a short-hand for saying p-value should be < 0.01
        assert res.pvalue < 0.001
    else:
        np.testing.assert_allclose(res.pvalue, expected.pvalue, rtol=3e-2)


@pytest.mark.parametrize(
    ("X", "expected"),
    [
        (sample1_royston, HenzeResult(0.497, 0.107)),
        (sample2, HenzeResult(1.377, 0.001)),
        (sample3_royston, HenzeResult(0.65, 0.305)),
        (sample4, HenzeResult(0.26, 0.602)),
    ],
)
def test_henze(X: np.ndarray, expected: HenzeResult):
    # call to henze
    res = henze(X)

    np.testing.assert_allclose(res.statistic, expected.statistic, rtol=1e-2)

    if expected.pvalue == 0.001:
        assert res.pvalue < 0.001
    else:
        np.testing.assert_allclose(res.pvalue, expected.pvalue, rtol=1e-2)


@pytest.mark.parametrize(
    ("X", "expected"),
    [
        (sample1_royston, DoornikResult(0.8337265, 0.9338702)),
        (sample2, DoornikResult(19.98276, 0.0005033284)),
        (sample3_royston, DoornikResult(10.74224, 0.2167362)),
        (sample4, DoornikResult(2.803507, 0.5912275)),
        # univariate tests
        # note that since mvnTest does not support the Doornik-Hansen test for univariate data
        # R's asbio package was used instead
        (sample_univariate, DoornikResult(0.6391224, 0.7264677)),
        (sample_univariate2, DoornikResult(3.258134, 0.1961125)),
        (sample_univariate3, DoornikResult(29.26717, 4.412802e-07)),
    ],
)
def test_doornik(X: np.ndarray, expected: DoornikResult):
    # call to doornik
    res = doornik(X)

    # EXTREMELY IMPORTANT NOTE: Testing was done against R's mvnTest package
    # for some weird reason the results were not lining up against R's MVN package
    # this seems to be because R's MVN package uses the original 1994 working paper
    # while mvnTest uses the 2008 revision
    # AnovaPy utilizes formulas from the 2008 revision
    # so mvnTest was used

    np.testing.assert_allclose(res.statistic, expected.statistic, rtol=1e-2)
    np.testing.assert_allclose(res.pvalue, expected.pvalue, rtol=1e-2)


@pytest.mark.parametrize(
    ("X", "r", "expected"),
    [
        (sample1, 1000, EnergyResult(0.51547, 0.756)),
        (sample2, 1000, EnergyResult(1.7773, 0.001)),
        (sample3, 1000, EnergyResult(0.89791, 0.409)),
        (sample4, 1000, EnergyResult(0.50436, 0.806)),
        (sample1, 500, EnergyResult(0.51547, 0.764)),
        # also test on univariate input
        (sample_univariate, 1000, EnergyResult(0.48262, 0.291)),
        (sample_univariate2, 1000, EnergyResult(1.0534, 0.013)),
        (sample_univariate3, 1000, EnergyResult(2.5508, 0.001)),
        (sample_univariate, 100, EnergyResult(0.48262, 0.25)),
    ],
)
def test_energy(X: np.ndarray, r: int, expected: EnergyResult):
    # call to energy
    res = energy(
        X, r=r, seed=42
    )  # we will keep the seed always at 42 for reproducability

    np.testing.assert_allclose(res.statistic, expected.statistic, rtol=1e-2)

    if expected.pvalue == 0.001:
        assert res.pvalue < 0.001
    else:
        # Allow a slightly wider tolerance for Monte Carlo simulation variance
        np.testing.assert_allclose(res.pvalue, expected.pvalue, rtol=0.15, atol=0.05)


@pytest.mark.parametrize(
    ("X", "expected"),
    [
        (
            np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 2]),
            ShapiroFranciaResult(statistic=0.95247, pvalue=0.5811),
        ),
        (sample_univariate, ShapiroFranciaResult(statistic=0.9811, pvalue=0.4029)),
        (sample_univariate2, ShapiroFranciaResult(statistic=0.96589, pvalue=0.05191)),
        (sample_univariate3, ShapiroFranciaResult(statistic=0.85491, pvalue=1.709e-05)),
    ],
)
def test_shapiro(X: np.ndarray, expected: ShapiroFranciaResult):
    # get result
    res = shapiro_francia(X)

    np.testing.assert_allclose(res.statistic, expected.statistic, rtol=1e-2)
    np.testing.assert_allclose(res.pvalue, expected.pvalue, rtol=1e-2)
