# Box's M-Test

MANOVA has two critical assumptions that needed to be tested with two different tests: one is the multivariate normality assumption (see [Multivariate Normality](./MultivariateNormality.md) for learning how to check normality in multivariate data with AnovaPy) and the other is homogeneity of covariance matrices across groups. AnovaPy lets you test both.

This section strictly features how to use Box's M-Test in AnovaPy to check for the second assumption. It is pretty easy overall, and the below example demonstrates how:

```python
from anovapy.manova import box
import numpy as np

samples = np.random.normal(
    size=(60, 2)
)  # in this case, there are 2 dependent variables
labels = np.repeat(
    ["A", "B"], 30
)  # this can also consist of numbers -- it doesn't matter in AnovaPy generally
np.random.shuffle(labels)

# now pass it on
res = box(samples, labels)

print(f"M-Statistic: {res.statistic}, p-value: {res.pvalue}")
```

That's it! `samples` in this case refers to our multivariate data (which must be a 2D array) and `labels` obviously refers to your group labels. As already stated in the comments, it doesn't really matter what is in `labels`; it could be strings or simply numbers, regardless, AnovaPy is designed to work with the data written in it.

Note that by default, AnovaPy will automatically apply a correction factor. This is to provide a more accurate p-value for small sample sizes (*note that Box's M remains sensitive to non-normality regardless, so always check multivariate normality first*). If you do not want to apply any correction, you can turn this off as follows:

```python
res_no_correction = box(samples, labels, correction=False)

print(
    f"M-Statistic: {res_no_correction.statistic}, p-value: {res_no_correction.pvalue}"
)
```

When no correction is applied, the "unadjusted" M-Statistic is returned.

Note though that it is usually recommended to keep the correction factor and most modern libraries, such as Pingouin (which this function was also tested against) will automatically apply this.

Regardless of whether the correction factor is applied, the p-value is approximated from the Chi-Square distribution.