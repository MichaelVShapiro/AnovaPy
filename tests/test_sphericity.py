# test sphericity logic

import numpy as np
import pandas as pd
import pingouin as pg  # we will test against this library

from anovapy.sphericity import john, mauchly


def test_mauchly_simple():
    # for handling simple test cases
    generator = np.random.default_rng(42)  # for reproducable results
    sample_a = generator.normal(size=40)
    sample_b = generator.normal(size=40)
    sample_c = generator.normal(size=40)

    df = pd.DataFrame(
        {"sample_a": sample_a, "sample_b": sample_b, "sample_c": sample_c}
    )

    anovapy_res = mauchly(sample_a, sample_b, sample_c)
    result_pg = pg.sphericity(data=df)

    np.testing.assert_allclose(anovapy_res.statistic, result_pg.W, rtol=1e-2)
    np.testing.assert_allclose(anovapy_res.pvalue, result_pg.pval, rtol=1e-2)


def test_mauchly_random():
    # this one is where we generate a whole bunch of random data and test directly against pingouin
    rng = np.random.default_rng()
    dist = np.array(["normal", "exp", "chi", "log"])  # a distribution to pick from
    for i in range(100):
        print(f"Testing Random #{i + 1}")
        # make some random data to throw in
        # test different lengths to see how both handle it!
        num_arrays = rng.integers(3, 8)
        size = rng.integers(10, 200)
        dist_pick = rng.choice(dist)
        if dist_pick == "normal":
            arrays = [
                rng.normal(loc=rng.integers(0, 10), scale=rng.integers(2, 6), size=size)
                for _ in range(num_arrays)
            ]
        elif dist_pick == "exp":
            arrays = [
                rng.exponential(scale=rng.integers(2, 6), size=size)
                for _ in range(num_arrays)
            ]
        elif dist_pick == "chi":
            arrays = [
                rng.chisquare(df=rng.integers(2, 10), size=size)
                for _ in range(num_arrays)
            ]
        else:
            # log
            arrays = [
                rng.logistic(
                    loc=rng.integers(0, 10), scale=rng.integers(2, 6), size=size
                )
                for _ in range(num_arrays)
            ]

        # store into pandas dataframe for pingouin as needed
        df = pd.DataFrame({f"{i}_arr": arrays[i] for i in range(len(arrays))})

        # now get results
        anovapy_res = mauchly(*arrays)
        pg_res = pg.sphericity(data=df)

        np.testing.assert_allclose(
            anovapy_res.statistic,
            pg_res.W,
            rtol=1e-2,
            err_msg=f"Failed on statistic, AnovaPy ({anovapy_res.statistic}), Pg ({pg_res.W})",
        )
        # go more tolerance on the p-value here
        # AnovaPy uses the standard approximation when computing the p-value
        # that is C = (n - 1) - ((2k^2 - k + 2) / 6k)
        # which is slightly different from pingouin what it uses internally, but the resulting p-values should still look super similar
        np.testing.assert_allclose(
            anovapy_res.pvalue,
            pg_res.pval,
            rtol=5e-2,
            atol=1e-2,
            err_msg=f"Failed on p-value, AnovaPy ({anovapy_res.pvalue}), Pg ({pg_res.pval})",
        )


def test_john_simple():
    # for handling simple test cases
    generator = np.random.default_rng(42)  # for reproducable results
    sample_a = generator.normal(size=40)
    sample_b = generator.normal(size=40)
    sample_c = generator.normal(size=40)

    df = pd.DataFrame(
        {"sample_a": sample_a, "sample_b": sample_b, "sample_c": sample_c}
    )

    anovapy_res = john(sample_a, sample_b, sample_c)
    result_pg = pg.sphericity(data=df, method="jns")

    # result_pg.W returns the V statistic unlike result_pg.chi2 which returns the T-statistic, which is exactly what we are looking for
    np.testing.assert_allclose(
        anovapy_res.statistic,
        result_pg.chi2,
        rtol=1e-2,
        err_msg=f"Failed on statistic, AnovaPy ({anovapy_res.statistic}), Pg ({result_pg.W})",
    )
    np.testing.assert_allclose(
        anovapy_res.pvalue,
        result_pg.pval,
        rtol=1e-2,
        err_msg=f"Failed on p-value, AnovaPy ({anovapy_res.pvalue}), Pg ({result_pg.pval})",
    )


def test_john_random():
    # this one is where we generate a whole bunch of random data and test directly against pingouin
    rng = np.random.default_rng()
    dist = np.array(["normal", "exp", "chi", "log"])  # a distribution to pick from
    for i in range(100):
        print(f"Testing Random #{i + 1}")
        # make some random data to throw in
        # test different lengths to see how both handle it!
        num_arrays = rng.integers(3, 8)
        size = rng.integers(10, 200)
        dist_pick = rng.choice(dist)
        if dist_pick == "normal":
            arrays = [
                rng.normal(loc=rng.integers(0, 10), scale=rng.integers(2, 6), size=size)
                for _ in range(num_arrays)
            ]
        elif dist_pick == "exp":
            arrays = [
                rng.exponential(scale=rng.integers(2, 6), size=size)
                for _ in range(num_arrays)
            ]
        elif dist_pick == "chi":
            arrays = [
                rng.chisquare(df=rng.integers(2, 10), size=size)
                for _ in range(num_arrays)
            ]
        else:
            # log
            arrays = [
                rng.logistic(
                    loc=rng.integers(0, 10), scale=rng.integers(2, 6), size=size
                )
                for _ in range(num_arrays)
            ]

        # store into pandas dataframe for pingouin as needed
        df = pd.DataFrame({f"{i}_arr": arrays[i] for i in range(len(arrays))})

        # now get results
        anovapy_res = john(*arrays)
        pg_res = pg.sphericity(data=df, method="jns")

        np.testing.assert_allclose(
            anovapy_res.statistic,
            pg_res.chi2,
            rtol=1e-2,
            err_msg=f"Failed on statistic, AnovaPy ({anovapy_res.statistic}), Pg ({pg_res.W})",
        )
        # go more tolerance on the p-value here
        # AnovaPy uses the standard approximation when computing the p-value
        # that is C = (n - 1) - ((2k^2 - k + 2) / 6k)
        # which is slightly different from pingouin what it uses internally, but the resulting p-values should still look super similar
        np.testing.assert_allclose(
            anovapy_res.pvalue,
            pg_res.pval,
            rtol=5e-2,
            atol=1e-2,
            err_msg=f"Failed on p-value, AnovaPy ({anovapy_res.pvalue}), Pg ({pg_res.pval})",
        )
