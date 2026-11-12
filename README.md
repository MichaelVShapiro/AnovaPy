# AnovaPy

[![CI](https://github.com/MichaelVShapiro/AnovaPy/actions/workflows/test.yml/badge.svg)](https://github.com/MichaelVShapiro/AnovaPy/actions/workflows/test.yml)
[![Python 3.9](https://img.shields.io/badge/python-3.9+-blue)](https://img.shields.io/badge/python-3.9+-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

AnovaPy is a lightweight Python library, built around SciPy and NumPy, that can calculate One-Way ANOVA, Two-Way ANOVA, and MANOVA, along with post-hoc tests and alternative tests when assumptions aren't met.

## Table of Contents

- [What is it?](#what-is-it)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## What is it?

Analysis of Variance (ANOVA) is a test regularly used by researchers. There are several different kinds such as One Way ANOVA, Two Way ANOVA, and MANOVA, just to name a few.

[SciPy](https://github.com/scipy/scipy) allows developers and researchers to compute the F-statistic and p-value from a One Way ANOVA (see `scipy.stats.f_oneway`) and even run the Tukey HSD post-hoc test in case the p-value proves significant (see `scipy.stats.tukey_hsd`). It also supports a few other tests in case assumptions of ANOVA (i.e. normality and homogeneity of variance) are not met. The simplicity of its API, where developers need only to pass in basic NumPy arrays, is what makes it really easy for anyone to learn fast.

However, the library is missing several other forms of ANOVA and critical tests to check assumptions for them too. Some libraries such as [statsmodels](https://github.com/statsmodels/statsmodels) solve this problem, but they are really difficult to work with, and especially for beginners, it may not be easy to grasp the R-like syntax.

This is where AnovaPy comes in. AnovaPy is a special library that implements "ANOVA related" tests (including Two Way ANOVA and MANOVA along with some alternatives) and all of its functions have a "SciPy feel" to them. By "SciPy feel," I mean accepting raw NumPy arrays as inputs and returning clean, intuitive result objects with direct dot-notation access to statistics and p-values -- no formula strings or DataFrame setups required.

Forget having to work with statsmodels's `ols` function or setting anything else up just to get your result. AnovaPy makes it easy where you only have to pass in NumPy arrays just like SciPy! Here is an example of how running a Two Way ANOVA in AnovaPy compares to that of statsmodels:

```python
# prepare data
import numpy as np

watering_frequency = np.repeat(["Daily", "Weekly"], 20)
sunlight_exposure = np.tile(np.repeat(["None", "Low", "Medium", "High"], 5), 2)
plant_height = np.random.normal(size=40)  # Random data for illustration
# statsmodels example
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd

df = pd.DataFrame(
    {
        "factor_a": watering_frequency,
        "factor_b": sunlight_exposure,
        "dependent": plant_height,
    }
)

# fit the model
model = ols(
    "dependent ~ C(factor_a) + C(factor_b) + C(factor_a):C(factor_b)", data=df
).fit()

# Generate the ANOVA table
anova_table = sm.stats.anova_lm(model, typ=1)
print(anova_table)
```

No more need to do that with AnovaPy! You get all the results you need in just one simple line of code:

```python
from anovapy.twoway import f_twoway  # Mimics SciPy naming conventions

res = f_twoway(
    factor_a=watering_frequency, factor_b=sunlight_exposure, dependent=plant_height
)  # easy!

print(f"F-Statistic for Factor A: {res.statistic_a}")
print(f"p-value for Factor A: {res.pvalue_a}")
print(f"F-Statistic for Factor B: {res.statistic_b}")
print(f"p-value for Factor B: {res.pvalue_b}")
print(f"F-Statistic for Interaction: {res.statistic_interaction}")
print(f"p-value for Interaction: {res.pvalue_interaction}")
```

That's it! This is the main magic behind AnovaPy: to minimize the learning curve and make it as easy to use as possible. If you know how to work with SciPy, then you already know how to work with AnovaPy, it is no different. Because AnovaPy computes tests directly from NumPy arrays without heavy model-fitting overhead, it is significantly faster than general-purpose modeling libraries for standard factorial designs.

*Note: AnovaPy is not a replacement for statsmodels or similar general-purpose modeling libraries; its scope strictly focuses on making ANOVA-related workflows intuitive.*

While I will leave the official documentation of AnovaPy (located under `docs/`) to list more examples of how to use different functions and name more tests implemented, here are some more tests built in as of v0.1 so you can get a better idea of what AnovaPy has and some more areas you can explore in the documentation:

- One Way Repeated Measures ANOVA (available via `anovapy.oneway.f_repeated`)
- Scheirer-Ray-Hare Test (available via `anovapy.twoway.scheirer`)
- Aligned Rank Transform (ART) ANOVA (available via `anovapy.twoway.art`)
- Tukey's HSD Post-Hoc for Two Way ANOVA (available via `anovapy.twoway.tukey_hsd`)
- Multivariate Analysis of Variance (available via `anovapy.manova.f_manova`)
- Permutation Multivariate Analysis of Variance (PERMANOVA -- available via `anovapy.manova.permanova`)
- Doornik-Hansen test for checking multivariate normality, a critical assumption of MANOVA (available via `anovapy.normal.doornik`)
- Mardia's Test for checking multivariate normality, a critical assumption of MANOVA (available via `anovapy.normal.mardia`)
- Mauchly's Test for checking the sphericity assumption of One Way Repeated Measures ANOVA (available via `anovapy.sphericity.mauchly`)

Feel free to explore the documentation of these functions, *and more,* to learn more about how to use them.

## Installation

To install AnovaPy, run:

```bash
pip install git+https://github.com/MichaelVShapiro/AnovaPy.git
```

Alternatively you can clone this repo and install locally for developmental purposes with `pip install -e .`

## Contributing

Any contributions are always welcome to AnovaPy! My goal is to have AnovaPy give as much support to ANOVA-related functions as possible, all with its own "SciPy feel" so any developer can quickly understand how to work with it. See `CONTRIBUTING.md` on ways you can contribute.

## License

AnovaPy is open source! It is licensed under MIT (see LICENSE file).