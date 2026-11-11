# tests two-way anova related tests

import numpy as np
import pytest

from anovapy.twoway import art, artc, f_twoway, scheirer, tukey_hsd

# note that expected values were run with R
# various R packages were used for making test cases such as ARTool and rcompanion

# for testing more complex data
# once again, numbers are generated via terminal randomly
# np.repeat([0, 1, 2, 3], 10)
complex_factor_a = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
    ]
)
# np.tile(np.repeat([0, 1], 5), 4)
complex_factor_b = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
    ]
)
# np.random.normal(loc=10, scale=3, size=40)
complex_dependent = np.array(
    [
        16.50696545,
        13.72115192,
        7.79938476,
        10.74309071,
        9.06927399,
        7.22734213,
        17.66024817,
        10.88925524,
        9.75008858,
        13.73838377,
        7.20880374,
        6.63978869,
        8.90851589,
        15.78309324,
        3.96084311,
        6.29747248,
        7.55166852,
        9.0553393,
        9.1010614,
        10.98034881,
        11.3942319,
        7.11888642,
        4.39234426,
        10.07522327,
        12.31790243,
        18.02386101,
        6.32329068,
        13.12519838,
        7.49690246,
        8.77906157,
        17.92689315,
        12.75246304,
        13.67819851,
        6.6284239,
        14.99648051,
        9.64348867,
        10.20408209,
        9.22225499,
        12.73334565,
        7.19616647,
    ]
)

basic_factor_a = np.array(["F1", "F1", "F1", "F2", "F2", "F2", "F3", "F3", "F3"])
basic_factor_b = np.array(
    [
        "Wheat",
        "Rice",
        "Wheat",
        "Rice",
        "Wheat",
        "Rice",
        "Wheat",
        "Rice",
        "Wheat",
    ]
)
basic_dependent = np.array([12.3, 14.5, 11.8, 16.1, 15.2, 17.0, 10.1, 11.2, 9.8])


def test_twoway_valid():
    # just reusing the same data as above but making it work for two way anova
    factor_a = np.array(
        ["F1", "F1", "F1", "F1", "F2", "F2", "F2", "F2", "F3", "F3", "F3", "F3"]
    )
    factor_b = np.array(
        [
            "Wheat",
            "Rice",
            "Wheat",
            "Rice",
            "Wheat",
            "Rice",
            "Wheat",
            "Rice",
            "Wheat",
            "Rice",
            "Wheat",
            "Rice",
        ]
    )
    dependent = np.array(
        [12.3, 14.5, 11.8, 16.1, 15.2, 17.0, 10.1, 11.2, 9.8, 10.2, 8.8, 7.8]
    )

    res = f_twoway(factor_a=factor_a, factor_b=factor_b, dependent=dependent)

    np.testing.assert_allclose(res.statistic_a, 4.44, rtol=1e-2)
    np.testing.assert_allclose(res.statistic_b, 1.12, rtol=1e-2)
    np.testing.assert_allclose(res.statistic_interaction, 0.546, rtol=1e-2)

    np.testing.assert_allclose(res.pvalue_a, 0.066, rtol=1e-2)
    np.testing.assert_allclose(res.pvalue_b, 0.331, rtol=1e-2)
    np.testing.assert_allclose(res.pvalue_interaction, 0.605, rtol=1e-2)


def test_tukey_valid():
    # just reusing the same data as above but making it work for two way anova
    factor_a = np.array(
        ["F1", "F1", "F1", "F1", "F2", "F2", "F2", "F2", "F3", "F3", "F3", "F3"]
    )
    factor_b = np.array(
        [
            "Wheat",
            "Rice",
            "Wheat",
            "Rice",
            "Wheat",
            "Rice",
            "Wheat",
            "Rice",
            "Wheat",
            "Rice",
            "Wheat",
            "Rice",
        ]
    )
    dependent = np.array(
        [12.3, 14.5, 11.8, 16.1, 15.2, 17.0, 10.1, 11.2, 9.8, 10.2, 8.8, 7.8]
    )

    res = tukey_hsd(factor_a, factor_b, dependent)

    # factor a
    np.testing.assert_allclose(
        res.statistic, np.array([0.2498, 3.7681, 3.5182]), rtol=1e-2
    )
    np.testing.assert_allclose(res.pvalue, np.array([0.983, 0.0828, 0.1037]), rtol=1e-2)

    # factor b
    res_b = tukey_hsd(factor_a, factor_b, dependent, term="factor_b")

    np.testing.assert_allclose(res_b.statistic, np.array([1.4958]), rtol=1e-2)
    np.testing.assert_allclose(res_b.pvalue, np.array([0.3309]), rtol=1e-2)

    # interaction
    res_int = tukey_hsd(factor_a, factor_b, dependent, term="interaction")

    expected = np.array(
        [
            0.70659583,
            3.7096281,
            1.91369704,
            1.56039912,
            3.53297914,
            3.00303227,
            1.20710121,
            0.85380329,
            2.82638332,
            1.79593107,
            2.14922898,
            0.17664896,
            0.35329791,
            1.61928211,
            1.97258002,
        ]
    )

    expected = np.sort(expected)
    res_int_sorted = np.sort(res_int.statistic)

    np.testing.assert_allclose(
        res_int_sorted,
        expected,
        rtol=1e-2,
    )

    expected = np.array(
        [
            0.99426606,
            0.22346834,
            0.75137216,
            0.86423475,
            0.25657938,
            0.38320758,
            0.94518476,
            0.98671666,
            0.43493788,
            0.79151243,
            0.66717531,
            0.99999296,
            0.9997852,
            0.84721983,
            0.73067725,
        ]
    )

    expected = np.sort(expected)
    res_int_sorted = np.sort(res_int.pvalue)

    np.testing.assert_allclose(
        res_int_sorted,
        expected,
        rtol=1e-2,
    )


@pytest.mark.skip(reason="artc (`interaction=True`) is not yet implemented")
def test_artc_interaction():
    res = artc(
        factor_a=basic_factor_a,
        factor_b=basic_factor_b,
        dependent=basic_dependent,
        term="interaction",
        interaction=True,
    )

    assert len(res) == 3

    np.testing.assert_allclose(res[0].statistic, 0.7, rtol=1e-2)
    np.testing.assert_allclose(res[0].p_value_adjusted, 0.5343, rtol=1e-2)

    np.testing.assert_allclose(res[1].statistic, 1.6, rtol=1e-2)
    np.testing.assert_allclose(res[1].p_value_adjusted, 0.2079, rtol=1e-2)

    np.testing.assert_allclose(res[2].statistic, 0.9, rtol=1e-2)
    np.testing.assert_allclose(res[2].p_value_adjusted, 0.4345, rtol=1e-2)


class TestTwoWayAnova:
    """
    All tests related to two-way anova
    """

    def setup_method(self):
        self.factor_a = np.array(["F1", "F1", "F1", "F2", "F2", "F2", "F3", "F3", "F3"])
        self.factor_b = np.array(
            [
                "Wheat",
                "Rice",
                "Wheat",
                "Rice",
                "Wheat",
                "Rice",
                "Wheat",
                "Rice",
                "Wheat",
            ]
        )
        self.dependent = np.array([12.3, 14.5, 11.8, 16.1, 15.2, 17.0, 10.1, 11.2, 9.8])

        # the random factors below were generated (by terminal) with np.random.poisson(lam=3, size=40)
        self.random_factor_a = np.array(
            [
                2,
                3,
                7,
                6,
                3,
                1,
                4,
                3,
                3,
                5,
                5,
                4,
                5,
                5,
                1,
                2,
                3,
                3,
                1,
                5,
                5,
                2,
                2,
                2,
                1,
                6,
                6,
                5,
                4,
                2,
                0,
                2,
                6,
                5,
                0,
                3,
                2,
                4,
                8,
                2,
            ]
        )
        self.random_factor_b = np.array(
            [
                4,
                2,
                4,
                5,
                5,
                1,
                2,
                4,
                5,
                4,
                0,
                6,
                2,
                3,
                4,
                4,
                5,
                3,
                2,
                3,
                4,
                1,
                5,
                2,
                4,
                5,
                4,
                5,
                5,
                3,
                2,
                2,
                2,
                1,
                5,
                3,
                2,
                2,
                3,
                0,
            ]
        )
        # the dependent was generated (by terminal) with np.random.exponential(scale=7, size=(40,))
        self.random_dependent = np.array(
            [
                7.35316193,
                6.69021284,
                11.43014557,
                7.64751836,
                10.34059214,
                4.90650882,
                1.9446979,
                0.06862008,
                12.5815375,
                12.17888584,
                12.38968936,
                9.3872058,
                3.84296664,
                2.59149535,
                0.94915158,
                12.19096426,
                22.74411692,
                2.80503979,
                1.09623789,
                27.8406756,
                4.87442127,
                5.48421666,
                1.39971137,
                0.76330423,
                1.28640226,
                1.2398039,
                1.94640992,
                1.21109931,
                12.310312,
                0.80019069,
                7.91163258,
                9.79744506,
                1.31090759,
                0.2844749,
                0.16621113,
                6.00697212,
                21.21223834,
                2.27839901,
                2.86063628,
                9.90645159,
            ]
        )

    def test_two_way_simple(self):
        # this test case is taken from: https://www.statology.org/two-way-anova-by-hand/
        # Recreating the article dataset
        water = np.repeat(["Daily", "Weekly"], 20)
        sun = np.tile(np.repeat(["None", "Low", "Medium", "High"], 5), 2)
        height = np.array(
            [
                # Daily
                4.8,
                4.4,
                3.2,
                3.9,
                4.4,  # None
                5.0,
                5.2,
                5.6,
                4.3,
                4.8,  # Low
                6.4,
                6.2,
                4.7,
                5.5,
                5.8,  # Medium
                6.3,
                6.4,
                5.6,
                4.8,
                5.8,  # High
                # Weekly
                4.4,
                4.2,
                3.8,
                3.7,
                3.9,  # None
                4.9,
                5.3,
                5.7,
                5.4,
                4.8,  # Low
                5.8,
                6.2,
                6.3,
                6.5,
                5.5,  # Medium
                6.0,
                4.9,
                4.6,
                5.6,
                5.5,  # High
            ]
        )
        res = f_twoway(factor_a=water, factor_b=sun, dependent=height)

        # degrees of freedom
        assert res.df_a == 1
        assert res.df_b == 3
        assert res.df_interaction == 3

        # f-statistic
        np.testing.assert_allclose(res.statistic_a, 0.000921, rtol=1e-2)
        np.testing.assert_allclose(res.statistic_b, 23.04898, rtol=1e-5)
        np.testing.assert_allclose(res.statistic_interaction, 1.241517, rtol=1e-6)

        # p-value
        np.testing.assert_allclose(res.pvalue_a, 0.975, rtol=1e-3)
        assert res.pvalue_b < 0.0001  # this is super small
        np.testing.assert_allclose(res.pvalue_interaction, 0.311, rtol=1e-3)

    def test_two_way_simple2(self):
        # for testing two way on more data
        # this one should throw an error though
        with pytest.raises(ValueError):
            res = f_twoway(
                factor_a=self.factor_a, factor_b=self.factor_b, dependent=self.dependent
            )

            # check up on results
            # we should never reach these

            # degrees of freedom
            assert res.df_a == 2
            assert res.df_b == 1
            assert res.df_interaction == 2
            assert res.df_error == 3

            # f-stat
            np.testing.assert_allclose(res.statistic_a, 129.328, rtol=1e-2)
            np.testing.assert_allclose(res.statistic_b, 29.568, rtol=1e-2)
            np.testing.assert_allclose(res.statistic_interaction, 1.542, rtol=1e-2)

            # p-vals
            np.testing.assert_allclose(res.pvalue_a, 0.0012276884, rtol=1e-4)
            np.testing.assert_allclose(res.pvalue_b, 0.01221, rtol=1e-3)
            np.testing.assert_allclose(res.pvalue_interaction, 0.34625, rtol=1e-3)

    def test_tukey(self):
        # test on tutorial data
        water = np.repeat(["Daily", "Weekly"], 20)
        sun = np.tile(np.repeat(["None", "Low", "Medium", "High"], 5), 2)
        height = np.array(
            [
                # Daily
                4.8,
                4.4,
                3.2,
                3.9,
                4.4,  # None
                5.0,
                5.2,
                5.6,
                4.3,
                4.8,  # Low
                6.4,
                6.2,
                4.7,
                5.5,
                5.8,  # Medium
                6.3,
                6.4,
                5.6,
                4.8,
                5.8,  # High
                # Weekly
                4.4,
                4.2,
                3.8,
                3.7,
                3.9,  # None
                4.9,
                5.3,
                5.7,
                5.4,
                4.8,  # Low
                5.8,
                6.2,
                6.3,
                6.5,
                5.5,  # Medium
                6.0,
                4.9,
                4.6,
                5.6,
                5.5,  # High
            ]
        )

        res_a = tukey_hsd(water, sun, height, term="factor_a")

        assert len(res_a.comparisons) == 1
        assert (
            res_a.comparisons[0][0] == "Daily" and res_a.comparisons[0][1] == "Weekly"
        )
        np.testing.assert_allclose(res_a.pvalue, np.array(0.975975), rtol=1e-2)

        # check factor b
        res_b = tukey_hsd(water, sun, height, term="factor_b")
        assert len(res_b.comparisons) == 6
        np.testing.assert_allclose(
            res_b.pvalue,
            np.array(
                [
                    0.2353814,
                    0.4730539,
                    0.0000023,
                    0.0095962,
                    0.0005862,
                    3.845520402112612e-08,
                ]
            ),
            rtol=1e-2,
        )

        # check interaction
        res_interaction = tukey_hsd(water, sun, height, term="interaction")
        assert len(res_interaction.comparisons) == 28
        expected_pvals = np.array(
            [
                0.8523023,
                0.2626201,
                0.6873529,
                0.9999996,
                0.9884511,
                0.0005069,
                0.0001516,
                0.9657630,
                0.9999854,
                0.9217373,
                0.3529276,
                0.0219109,
                0.0073419,
                0.9954055,
                0.3529276,
                0.0456827,
                0.2118374,
                0.0905656,
                0.7925032,
                0.2118374,
                0.0456827,
                0.0161418,
                0.9657630,
                0.0008466,
                0.0002546,
                0.0000450,
                0.0000134,
                0.9998578,
            ]
        )
        expected_pvals = np.sort(expected_pvals)
        res_int = res_interaction.pvalue
        res_int = np.sort(res_int)
        np.testing.assert_allclose(res_int, expected_pvals, rtol=1e-2)

    def test_tukey2(self):
        with pytest.raises(ValueError):
            res = tukey_hsd(self.factor_a, self.factor_b, self.dependent)

            # this should never reach
            print(f"{res.comparisons}")

    def test_scheirer_simple(self):
        # test super simple and lightweight calculations
        # note that 'fertilizer' and 'crop' doesn't have to hold string data, it can also hold numerical data!
        fertilizer = np.array(["F1", "F1", "F1", "F2", "F2", "F2", "F3", "F3", "F3"])
        crop = np.array(
            [
                "Wheat",
                "Rice",
                "Wheat",
                "Rice",
                "Wheat",
                "Rice",
                "Wheat",
                "Rice",
                "Wheat",
            ]
        )
        yield_amount = np.array([12.3, 14.5, 11.8, 16.1, 15.2, 17.0, 10.1, 11.2, 9.8])

        res = scheirer(factor_a=fertilizer, factor_b=crop, dependent=yield_amount)

        # check degrees of freedom
        assert res.df_a == 2
        assert res.df_b == 1
        assert res.df_interaction == 2

        # use numpy for testing values
        # check h-statistics (up to 4 decimal places)
        check = np.array([res.statistic_a, res.statistic_b, res.statistic_interaction])
        np.testing.assert_allclose(check, np.array([5.64, 0.6, 0.0]), atol=1e-4)

        # check p-values same way
        check = np.array([res.pvalue_a, res.pvalue_b, res.pvalue_interaction])
        np.testing.assert_allclose(check, np.array([0.05961, 0.43858, 1.0]), atol=1e-4)

    def test_scheirer_complex(self):
        # just some more complex data
        res = scheirer(
            factor_a=self.random_factor_a,
            factor_b=self.random_factor_b,
            dependent=self.random_dependent,
        )

        assert res.df_a == 8
        assert res.df_b == 6
        assert res.df_interaction == 14

        # check stats
        check = np.array([res.statistic_a, res.statistic_b, res.statistic_interaction])
        np.testing.assert_allclose(
            check, np.array([5.4637, 2.9028, 20.6720]), rtol=1e-4
        )

        # check p-values
        check = np.array([res.pvalue_a, res.pvalue_b, res.pvalue_interaction])
        np.testing.assert_allclose(
            check, np.array([0.70706, 0.82094, 0.11034]), rtol=1e-4
        )

    def test_art_simple(self):
        # for testing basic ART ANOVA
        res = art(self.factor_a, self.factor_b, self.dependent)

        # validate each factor
        assert res.df_num_a == 2
        assert res.df_den_a == 3
        np.testing.assert_allclose(res.statistic_a, 12, rtol=1e-4)
        np.testing.assert_allclose(res.pvalue_a, 0.037037, rtol=1e-4)

        assert res.df_num_b == 1
        assert res.df_den_b == 3
        np.testing.assert_allclose(res.statistic_b, 8.3793, rtol=1e-4)
        np.testing.assert_allclose(res.pvalue_b, 0.062772, rtol=1e-4)

        # Note: The below assertion diverges slightly from R's ARTool output due to
        # floating-point precision differences. R/ARTool evaluates or rounds intermediate
        # values around the 8th decimal place, whereas AnovaPy utilizes NumPy
        # `float64` precision. At extreme boundary conditions (for instance, in this case, it divergence at the 15th
        # decimal place), this alters rankdata tie-breaking. AnovaPy's implementation
        # follows the official formula precisely using full double-precision memory.
        # The below values show values ARTool printed out for the interaction
        # assert res.interaction.df_num == 2
        # assert res.interaction.df_den == 3
        # np.testing.assert_allclose(res.interaction.statistic, 1.2867, rtol=1e-4)
        # np.testing.assert_allclose(res.interaction.pvalue, 0.394920, rtol=1e-4)

    def test_art_complex(self):

        res = art(
            factor_a=complex_factor_a,
            factor_b=complex_factor_b,
            dependent=complex_dependent,
        )

        assert res.df_num_a == 3 and res.df_den_a == 32
        np.testing.assert_allclose(res.statistic_a, 1.85812, rtol=1e-4)
        np.testing.assert_allclose(res.pvalue_a, 0.15658, rtol=1e-4)

        assert res.df_num_b == 1 and res.df_den_b == 32
        np.testing.assert_allclose(res.statistic_b, 0.24375, rtol=1e-4)
        np.testing.assert_allclose(res.pvalue_b, 0.62488, rtol=1e-4)

        assert res.df_num_interaction == 3 and res.df_den_interaction == 32
        np.testing.assert_allclose(res.statistic_interaction, 0.80754, rtol=1e-4)
        np.testing.assert_allclose(res.pvalue_interaction, 0.49904, rtol=1e-4)

    def test_art_post_hoc(self):
        # this test case explicitly tests the post hoc
        res = artc(self.factor_a, self.factor_b, self.dependent, term="factor_a")

        # there should be exactly three contrasts
        assert len(res.comparisons) == 3

        # test each contrast
        assert res.comparisons[0][0] == "F1" and res.comparisons[0][1] == "F2"
        assert res.df[0] == 3
        np.testing.assert_allclose(res.statistic[0], -2.4494897428, rtol=1e-4)
        np.testing.assert_allclose(res.pvalue[0], 0.1834422266, rtol=1e-4)

        assert res.comparisons[1][0] == "F1" and res.comparisons[1][1] == "F3"
        assert res.df[1] == 3
        np.testing.assert_allclose(res.statistic[1], 2.4494897428, rtol=1e-4)
        np.testing.assert_allclose(res.pvalue[1], 0.1834422266, rtol=1e-4)

        assert res.comparisons[2][0] == "F2" and res.comparisons[2][1] == "F3"
        assert res.df[2] == 3
        np.testing.assert_allclose(res.statistic[2], 4.8989794856, rtol=1e-4)
        np.testing.assert_allclose(res.pvalue[2], 0.0488298104, rtol=1e-4)

    def test_art_post_hoc2(self):
        # just test a different term
        res = artc(self.factor_a, self.factor_b, self.dependent, term="factor_b")

        # there should be exactly three contrasts
        assert len(res.comparisons) == 1

        # test each contrast
        # assert res[0].comparison == "R - W"
        assert res.df[0] == 3

        np.testing.assert_allclose(res.statistic[0], 2.895, rtol=1e-2)
        np.testing.assert_allclose(res.pvalue[0], 0.0628, rtol=1e-2)

    def test_art_post_hoc3(self):
        # tests the interaction
        res = artc(self.factor_a, self.factor_b, self.dependent, term="interaction")

        assert len(res.comparisons) == 15

        # to get to the point only test df and stats + pvalue
        # all df should be 3
        for i in res.df:
            assert i == 3

        # check stat
        expected_stats = np.array(
            [
                1.732,
                -2.887,
                -1.0,
                3.0,
                5.196,
                -5.657,
                -2.887,
                1.732,
                4.243,
                1.732,
                6.351,
                9.899,
                4.0,
                6.351,
                1.732,
            ]
        )
        given_stats = res.statistic
        np.testing.assert_allclose(np.array(given_stats), expected_stats, rtol=1e-3)

        # check pvalues
        expected_pvals = np.array(
            [
                0.9085,
                0.4614,
                0.9085,
                0.4614,
                0.1523,
                0.1313,
                0.4614,
                0.9085,
                0.2398,
                0.9085,
                0.1106,
                0.0329,
                0.2521,
                0.1106,
                0.9085,
            ]
        )
        given_vals = res.pvalue
        np.testing.assert_allclose(np.array(given_vals), expected_pvals, rtol=1e-2)
