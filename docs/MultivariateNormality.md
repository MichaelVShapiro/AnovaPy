# Multivariate Normality

One of MANOVA's assumptions is that data should be normally distributed. More specifically, one would check normality for each dependent variable within each group of categorical independent variables.

There are a few ways to check normality with AnovaPy, and this document covers all ways implemented.

## Table of Contents

- [Using Shapiro-Wilk Test](#using-shapiro-wilk-test)
- [Using Shapiro-Francia Test](#using-shapiro-francia-test)
- [Mardia's Test](#mardias-test)
- [Royston's Extension to Shapiro-Wilk](#roystons-extension-to-shapiro-wilk)
- [Henze-Zirkler Test](#henze-zirkler-test)
- [Doornik-Hansen Test](#doornik-hansen-test)
- [Energy Test (E-Statistic)](#energy-test-e-statistic)

## Using Shapiro-Wilk Test

One simple way to check normality is simply checking each dependent variable separately within each group using the Shapiro-Wilk Test. Since SciPy already implements this logic, you can easily make a call to `scipy.stats.shapiro`.

## Using Shapiro-Francia Test

Another way to check normality in univariate data (also an important of ANOVA generally) is the Shapiro-Francia test. *Note: for a full difference of what makes this test different from Shapiro-Wilk, see the doc string of `shapiro_francia`.* This can be achieved with the `shapiro_francia` function:

```python
from anovapy.normal import shapiro_francia
import numpy as np

# sample data
data = np.random.normal(size=50)

res = shapiro_francia(data)

print(f"W'-Statistic: {res.statistic}, p-value: {res.pvalue}")
```

## Mardia's Test

A very common test run on multivariate data to check normality is Mardia's Test. The example below illustrates how to make calls to the function:

```python
from anovapy.normal import mardia
import numpy as np

# note that since we are working with multivariate data, a 2D array must be passed in
samples = np.random.normal(loc=0.0, scale=1.0, size=(60, 80))

mardia_test = mardia(samples)

print(f"Statistic (Skew): {mardia_test.skew_stat}")
print(f"p-value (Skew): {mardia_test.skew_pvalue}")
print(f"Statistic (Kurtosis): {mardia_test.kurtosis_stat}")
print(f"p-value (Kurtosis): {mardia_test.kurtosis_pvalue}")
```

Note that AnovaPy was designed to have functions whose signatures are similar to SciPy's so there is no learning curve required. For instance, to handle NaN values, you may use the `nan_policy` parameter which functions identically to how other functions handle it in SciPy:

```python
from anovapy.normal import mardia
import numpy as np

samples = np.random.normal(loc=0.0, scale=1.0, size=(60, 80))

samples[0, 1] = np.nan  # just to keep it simple

# Propagating Results

mardia_test_propagate = mardia(samples, nan_policy="propagate")  # this is the default
print(f"Statistic should be np.nan: {mardia_test_propagate.statistic}")

# Raising Error

try:
    _ = mardia(samples, nan_policy="raise")
    print("will never reach")
except:
    print("error raised as needed")

# Omitting Values

res = mardia(samples, nan_policy="omit")
```

## Royston's Extension to Shapiro-Wilk

Another popular way to check normality in multivariate data before running MANOVA is to use Royston's Extension to Shapiro-Wilk. The simplest way to call it from AnovaPy is shown below:

```python
from anovapy.normal import royston
import numpy as np

samples = np.random.normal(loc=0.0, scale=1.0, size=(60, 80))

statistic, p_value = royston(samples)

print(f"H-statistic: {statistic}")
print(f"p-value: {p_value}")
```

A word on the implementation. The function is designed as according to the original paper. However, while testing against R's MVN package, a few differences were found. Looking into the source code, it seems how when the kurtosis of the dataset exceeds 3, the W-statistic from Shapiro-Francia test is used to help compute the H-statistic. The default behavior of the test implemented in AnovaPy is designed to match R's output so all test cases are consistent. To run Royston's Test using only the Shapiro-Wilk W-statistic, just run:

```python
royston_res = royston(
    samples, method="shapiro"
)  # forces to run the function as defined in the original paper

print(f"H-statistic: {royston_res.statistic}")
print(f"p-value: {royston_res.pvalue}")
```

## Henze-Zirkler Test

AnovaPy also features the Henze-Zirkler Test to check normality of multivariate data. The function signature remains the same and can be used as follows:

```python
from anovapy.normal import henze

henze_res = henze(samples)

print(f"HZ-Stat: {henze_res.statistic}")
print(f"p-value: {henze_res.pvalue}")
```

## Doornik-Hansen Test

AnovaPy also features the ability to check normality in multivariate data with the Doornik-Hansen Test. Like other functions mentioned herein, it contains the same signature and relatively easy to use:

```python
from anovapy.normal import doornik

doornik_res = doornik(samples)

print(f"Statistic: {doornik_res.statistic}")
print(f"p-value: {doornik_res.pvalue}")
```

As it turns out, while the Doornik-Hansen test can be used to check normality in multivariate data, it can also be used to check normality in univariate data! The `doornik` function in AnovaPy is designed to accept input that is 1D (univariate) or 2D (multivariate). The example above demonstrated how to work with multivariate (reusing `samples` of course). Working with 1D data is no different:

```python
rng = np.random.default_rng()

univariate_data = rng.normal(size=100)

doornik_1d_res = doornik(univariate_data)

print(f"Statistic: {doornik_1d_res.statistic}")
print(f"p-value: {doornik_1d_res.pvalue}")
```

The `doornik` function accepts either a one-dimensional array for a single variable or a two-dimensional array for multiple variables. Both forms use the same Doornik-Hansen moment transformations, with the approximate p-value calculated from a chi-square distribution with two degrees of freedom per variable.

## Energy Test (E-Statistic)

One last multivariate normality test AnovaPy features is the Energy Test (or rather getting the *E-Statistic*). Note that just like the Doornik-Hansen test, this test **also supports univariate data,** and hence either a 2D NumPy array (multivariate) or 1D NumPy array (univariate) can be passed in. Overall, the use of the function is straightforward:

```python
from anovapy.normal import energy

energy_res = energy(samples)

print(f"E-Statistic: {energy_res.statistic}")
print(f"p-value: {energy_res.pvalue}")

# or with univariate data
# note that we reuse the univariate_data variable made in the Doornik-Hansen test earlier

energy_res_1d = energy(univariate_data)

print(f"(Univariate Example) E-Statistic: {energy_res_1d.statistic}")
print(f"(Univariate Example) p-value: {energy_res_1d.pvalue}")
```

There are also some parameters you can mess around with in the `energy` function. More specifically the `r` parameter which controls the bootstrap resampling (this helps compute the p-value) and the `seed` parameter which also helps compute the p-value in short. When left out, `r` defaults to `1000` and `seed` defaults to `None`. The example below demonstrates how to work with both:

```python
res = energy(samples, r=500, seed=42)
```

For more info about the `r` and `seed` parameters check out the doc string of the `energy` function which explains their use case more in depth.