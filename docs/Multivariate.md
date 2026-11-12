# Multivariate Tests

AnovaPy implements Multiple Analysis of Variance (MANOVA) including a few "alternative" tests (in case assumptions aren't met). *Note: AnovaPy also implements tests to check normality in multivariate data -- a key assumption of MANOVA. See **MultivariateNormality.md** for more info on tests implemented and how to use them.*

## Table of Contents

- [MANOVA](#manova)
- [PERMANOVA](#permanova)
- [Multivariate Kruskal-Wallis](#multivariate-kruskal-wallis)
- [Non-Parametric MANOVA](#non-parametric-manova)
- [Transforming Dependent Variables](#transforming-dependent-variables)

## MANOVA

To run MANOVA, call the `f_manova` function in AnovaPy as shown below:

```python
from anovapy.manova import f_manova
import numpy as np

# just generate some random data
observations = np.random.normal(size=(60, 4))  # 4 dependent variables in this example
labels = np.repeat([0, 1, 2], 20)  # example labels
np.random.shuffle(labels)  # just to randomize spots

manova_res = f_manova(observations, labels)

print(f"Statistic: {manova_res.statistic}")
print(f"p-value: {manova_res.pvalue}")
```

That's it! As written directly in the doc string of the function:

> The recommended form is `f_manova(data, labels)`: `data` contains one observation per row and one dependent variable per column, while `labels` contains one categorical group label per observation. A 1D `data` array is treated as a single dependent variable. The number of groups is inferred from the distinct labels; each label corresponds to one independent sample.

Various multivariate test statistics can also be performed right on this function as shown below:

```python
wilks_res = f_manova(observations, labels, test="wilks")
pillai_res = f_manova(observations, labels, test="pillai")
hotelling_lawley_res = f_manova(observations, labels, test="hotelling_lawley")
roy_res = f_manova(observations, labels, test="roy")
```

Note that unlike One Way ANOVA and Two Way ANOVA, MANOVA does not have a post-hoc test.

## PERMANOVA

A popular test used when assumptions of MANOVA are not met is the Permutational MANOVA (PERMANOVA). This is built into AnovaPy and can be called with the function `permanova`. The simplest way to use it is shown below:

```python
from anovapy.manova import permanova

permanova_res = permanova(observations, labels)

print(f"Statistic: {permanova_res.statistic}")
print(f"p-value: {permanova_res.pvalue}")
```

You can also pass two group matrices directly. Other parameters include the number of permutations (via `permutations`, which defaults to `999`), the distance metric (via `distance_metric`, which defaults to `euclidean` and accepts anything supported by `scipy.spatial.distance.pdist`, [see the official documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html)), and a random state for reproducible p-values (an integer or NumPy generator):

```python
diff_perm = permanova(
    observations, labels, permutations=100
)  # setting number of permutations to do to compute p-value

other_distance_metric = permanova(
    observations, labels, distance_metric="jaccard"
)  # setting another distance to use

other_random_state = permanova(observations, labels, random_state=42)
```

## Multivariate Kruskal-Wallis

First off, let's make something extremely clear. **There is no single "universally agreed" upon formula for the Multivariate Kruskal-Wallis test.** Different authors handle cross-product matrices, ranking structures, and asymptotic chi-square vs. F-approximations differently. To give researchers and developers maximum flexibility, AnovaPy provides both the Katz & McSweeney (1980) rank-sum approach, grounded in the foundational rank-order theory of Puri & Sen (1969) (see full list of references in `multivariate_kruskal`) and the modern rank-based MANOVA approach (highlighted in the next section, also see `nonparametric_manova`), allowing you to choose the statistical framing best suited for your study.

The example below demonstrates how to run the Multivariate Kruskal-Wallis in AnovaPy:

```python
from anovapy.manova import multivariate_kruskal

# just generate some new fresh data
dependent_vars = np.random.normal(
    loc=10, scale=20, size=(40, 2)
)  # so basically there are two dependent variables in this example
independent_vars = np.repeat(["A", "B", "C", "D"], 10)
np.random.shuffle(independent_vars)  # shuffle in place

res = multivariate_kruskal(dependent_vars, independent_vars)

print(f"L-Statistic: {res.statistic}")
print(f"p-value: {res.pvalue}")
```

## Non-Parametric MANOVA

Finally, AnovaPy also lets you run a Non-Parametric MANOVA based on formulas established by Burchett et al. (2017) (see `nonparametric_manova` doc string for full reference). The below shows how:

```python
from anovapy.manova import nonparametric_manova

res = nonparametric_manova(dependent_vars, independent_vars)

print(f"F-Statistic: {res.statistic}")
print(f"p-value: {res.pvalue}")
```

## Transforming Dependent Variables

Like with Two Way ANOVA, in case assumptions are not met, you can also transform the dependent variables. This can be achieved as follows:

```python
from anovapy.manova import transform_dependent

transformed_vars = transform_dependent(dependent_vars, method="log")

# now run manova on transformed dependent variables
res = f_manova(transformed_vars, independent_vars)

# or provide a different method for transformation
transformed_sqrt = transform_dependent(dependent_vars, method="sqrt")

transformed_boxcox = transform_dependent(dependent_vars, method="boxcox")
```

After transformation, **always recheck your assumptions** before running `f_manova` again.