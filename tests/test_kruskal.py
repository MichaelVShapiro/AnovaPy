# for testing multivariate kruskal wallis test
# this file also tests the Non-Parametric MANOVA

# note that everything here was tested against R's npmv package (Non-Parametric MANOVA)
# and tested against ULT (which contains the function for the Multivariate Kruskal-Wallis)

import numpy as np
import pytest

from anovapy.manova.kruskal import MultivariateKruskalResult, multivariate_kruskal
from anovapy.manova.nonparametric import NonParametricManovaResult, nonparametric_manova

# expected df = 4
simple_dependent_data = np.array(
    [
        [2.5, 48.0],
        [1.2, 25.5],
        [3.8, 30.1],
        [4.1, 42.0],
        [2.0, 53.1],
        [5.0, 22.1],
        [5.3, 50.2],
        [4.8, 28.0],
        [1.0, 40.1],
        [2.2, 29.5],
    ]
)

simple_independent_data = np.array(["A", "A", "B", "B", "A", "C", "C", "C", "B", "A"])

# the equivalent of representing the above in R (for this test case) is:
# library(npmv)
# my_data <- data.frame(
#  y1 = c(2.5, 1.2, 3.8, 4.1, 2.0, 5.0, 5.3, 4.8, 1.0, 2.2),
#  y2 = c(48.0, 25.5, 30.1, 42.0, 53.1, 22.1, 50.2, 28.0, 40.1, 29.5),
#  Group = factor(c('A', 'A', 'B', 'B', 'A', 'C', 'C', 'C', 'B', 'A'))
# )
# test_result <- nonpartest(y1 | y2 ~ Group, data = my_data)

# for the ULT package, it would look like:
# library(ULT)
# test_result <- multkw(group = my_data$Group, y = my_data[, c("y1", "y2")])

# also very important note: R's npmv package usually outputs some more statistics and p-values such as Wilk's Lambda
# at the current stage of AnovaPy though, the Non-Parametric MANOVA function doesn't have such functionality to output other statistics, it is strictly limited to Lawley-Hotelling Trace along with its adjustment to the F-statistic

# generated via np.random.normal(loc=10, scale=50, size=(20, 2))
complex_dependent = np.array(
    [
        [-6.00329157, 36.57942197],
        [-66.68294439, 34.08463605],
        [65.25305189, -36.18089619],
        [22.47959832, 9.43111094],
        [-34.35726453, 21.66030825],
        [35.33907971, 9.86767822],
        [-52.59770233, 11.82678376],
        [53.76167259, -22.95405414],
        [20.09619342, -44.35642318],
        [-7.62307458, 51.32106235],
        [95.42752655, 21.60407136],
        [44.1198796, 41.32149319],
        [-33.47590456, -17.4238744],
        [-63.20450777, 8.54752992],
        [-19.94209217, 27.67434589],
        [-37.80367991, -14.09336858],
        [111.31451314, 12.64825652],
        [22.77636737, 59.45218713],
        [54.77384014, -39.55919139],
        [-85.46965322, -2.7384823],
    ]
)
complex_independent = np.array(
    [
        "A",
        "A",
        "A",
        "A",
        "A",
        "B",
        "B",
        "B",
        "B",
        "B",
        "C",
        "C",
        "C",
        "C",
        "C",
        "D",
        "D",
        "D",
        "D",
        "D",
    ]
)

complex_dependent2 = np.array(
    [
        [-91.98795658, -121.88455733, 77.5760623],
        [101.56077729, 88.87372075, 22.76558371],
        [-4.09137045, 169.96643632, -106.24365231],
        [59.38507155, 22.44312168, 184.66845296],
        [68.20929805, 47.3262612, 27.17098162],
        [10.16214156, 98.5093247, -47.44876538],
        [140.43171868, 21.55983534, 50.7716949],
        [54.52864734, -27.79053361, 13.70407351],
        [46.39008368, 143.21253811, 29.49847609],
        [46.45830724, 82.4579623, -155.66175613],
        [-9.17143626, 48.55279279, 188.58100935],
        [109.63089484, 228.05434992, 14.02516062],
        [135.4770535, -158.15388076, 53.49255092],
        [68.518263, 117.00973348, -90.91309689],
        [110.89186789, -39.6698726, 12.30805087],
        [63.07108412, 141.60568108, 43.51101617],
        [10.75485906, -96.32815747, 33.45644446],
        [-96.01218723, 246.63548623, 107.92798493],
        [199.57482587, 170.78598482, -20.4962651],
        [90.16542273, 139.69579759, 95.69175356],
        [240.84970128, 121.09127907, -139.12328014],
        [119.60943514, -1.78426913, 8.20038704],
        [35.2018973, -127.25443301, 55.54629648],
        [122.51325306, -74.21620923, 14.59239233],
        [34.32261885, 169.7492657, 101.13990314],
        [-23.38970219, 241.87588149, 49.30349583],
        [-16.99763163, 395.51277247, 31.50362909],
        [177.6416993, -103.9840259, 79.25636185],
        [114.97172149, -13.85212606, 170.93224499],
        [-115.7543404, 119.55717962, -60.89559563],
    ]
)
complex_independent2 = np.array(
    [
        "C",
        "B",
        "C",
        "A",
        "B",
        "A",
        "C",
        "A",
        "C",
        "A",
        "A",
        "C",
        "B",
        "C",
        "C",
        "A",
        "A",
        "A",
        "B",
        "B",
        "C",
        "B",
        "C",
        "A",
        "A",
        "B",
        "B",
        "B",
        "B",
        "C",
    ]
)


@pytest.mark.parametrize(
    ("independent", "dependent", "expected"),
    [
        (
            simple_independent_data,
            simple_dependent_data,
            NonParametricManovaResult(statistic=3.616, pvalue=0.075),
        ),
        (
            complex_independent,
            complex_dependent,
            NonParametricManovaResult(statistic=0.152, pvalue=0.986),
        ),
        (
            complex_independent2,
            complex_dependent2,
            NonParametricManovaResult(statistic=1.391, pvalue=0.249),
        ),
    ],
)
def test_nonparametric_manova(
    independent, dependent, expected: NonParametricManovaResult
):
    # get result from anovapy
    res = nonparametric_manova(dependent, independent)

    # check
    np.testing.assert_allclose(res.statistic, expected.statistic, rtol=0, atol=5e-4)
    np.testing.assert_allclose(res.pvalue, expected.pvalue, rtol=0, atol=5e-4)


@pytest.mark.parametrize(
    ("independent", "dependent", "expected"),
    [
        (
            simple_independent_data,
            simple_dependent_data,
            MultivariateKruskalResult(statistic=6.429045, pvalue=0.1693164),
        ),
        (
            complex_independent,
            complex_dependent,
            MultivariateKruskalResult(statistic=1.121309, pvalue=0.9805931),
        ),
        (
            complex_independent2,
            complex_dependent2,
            MultivariateKruskalResult(statistic=8.295561, pvalue=0.2172395),
        ),
    ],
)
def test_kruskal(independent, dependent, expected: NonParametricManovaResult):
    # get result from anovapy
    res = multivariate_kruskal(dependent, independent)

    # check
    np.testing.assert_allclose(res.statistic, expected.statistic, rtol=0, atol=5e-4)
    np.testing.assert_allclose(res.pvalue, expected.pvalue, rtol=0, atol=5e-4)
