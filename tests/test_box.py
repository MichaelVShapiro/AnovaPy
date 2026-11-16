# for testing Box-M test

# we will test against pingouin here
import numpy as np
import pandas as pd
import pingouin as pg

from anovapy.manova import box

# run against a whole bunch of random tests


def test_against_pg():
    for i in range(100):
        print(f"Test #{i}")

        # generate random data of random length
        p = np.random.randint(2, 6)  # number of dependent variables
        n_per_group = np.random.randint(10, 120)
        n_total = 2 * n_per_group  # 2 groups, equal size

        # each column is a 1D array of length n_total
        d = {
            f"var_{j}": np.random.normal(loc=3 * j, scale=40, size=n_total)
            for j in range(p)
        }
        d["group"] = np.repeat(["A", "B"], n_per_group)
        df = pd.DataFrame(d)

        # pingouin
        res_p = pg.box_m(df, dvs=[f"var_{j}" for j in range(p)], group="group")

        # remove the label group
        labels = df["group"].values
        df = df.drop(columns=["group"])

        # now AnovaPy
        res_anovapy = box(df.to_numpy(), labels)

        np.testing.assert_allclose(
            res_anovapy.statistic, res_p["Chi2"].values[0], rtol=1e-2
        )
        np.testing.assert_allclose(
            res_anovapy.pvalue, res_p["pval"].values[0], rtol=1e-2
        )
