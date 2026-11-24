# One Way Tests

Generally speaking, SciPy already implements One Way ANOVA via the `f_oneway` function (see `scipy.stats.f_oneway`). It also implements "alternative" tests (in case assumptions aren't met).

For this reason, there aren't many tests AnovaPy implements at the moment regarding One Way ANOVA and other related non-parametric tests.

The below table of contents summarizes all ANOVA-related tests supported by AnovaPy.

## Table of Contents

- [Dunn's Post-Hoc](#dunns-post-hoc)
- [Brown-Forsythe ANOVA Test](#brown-forsythe-anova-test)
- [One Way Repeated Measures ANOVA](#one-way-repeated-measures-anova)
- [One Way ANCOVA](#one-way-ancova)
- [ART One Way ANOVA](#art-one-way-anova)

## Dunn's Post-Hoc

The below example demonstrates how to work with Dunn's post-hoc test which is usually done after running the Kruskal-Wallis test (see `scipy.stats.kruskal`).

```python
from anovapy.oneway import dunns
from scipy.stats import kruskal
import numpy as np

# this example will keep things super simple
# just three samples is all we will work with
sample1 = np.random.normal(size=60)
sample2 = np.random.normal(size=60)
sample3 = np.random.normal(size=60)

# typically you will run the Kruskal-Wallis test first
kruskal_res = kruskal(sample1, sample2, sample3)

# check if result is statisitcally significant
if kruskal_res.pvalue < 0.05:
    # significant!
    # now run Dunn's post hoc
    dunns_result = dunns(sample1, sample2, sample3)

    for i in range(len(dunns_result.statistic)):
        group1, group2 = dunns_result.comparisons[i]
        print(
            f"Comparison between groups ({group1}, {group2}), statistic: {dunns_result.statistic[i]}, p-value: {dunns_result.pvalue[i]}"
        )
```

Note that the `dunns` implementation in AnovaPy also gives you the ability to change how the adjusted p-value is computed:

```python
dunns_res_bonferroni = dunns(sample1, sample2, sample3, adjust="bonferroni")
dunns_res_holm = dunns(sample1, sample2, sample3, adjust="holm")
dunns_res_sidak = dunns(sample1, sample2, sample3, adjust="sidak")
dunns_res_benjamini_hochberg = dunns(sample1, sample2, sample3, adjust="benjamini")
dunns_res_no_adjust = dunns(sample1, sample2, sample3, adjust="none")
```

The default value assigned to `adjust` is `bonferroni`.

## Brown-Forsythe ANOVA Test

You can also work with the Brown-Forsythe ANOVA test as follows:

```python
from anovapy.oneway import brown

# reusing sample1, sample2, and sample3 from the previous example

brown_res = brown(sample1, sample2, sample3)

print(f"F-Statistic: {brown_res.statistic}, p-value: {brown_res.pvalue}")
```

## One Way Repeated Measures ANOVA

A repeated measures ANOVA is used to determine whether or not there is a statistically significant difference between the means of three or more groups in which the same subjects show up in each group.

AnovaPy let's you run a One Way Repeated Measures ANOVA via the `anovapy.oneway.f_repeated` function as shown below:

```python
from anovapy.oneway import f_repeated

# we will resuse sample1, sample2, and sample3 again for simplicity

res = f_repeated(sample1, sample2, sample3)

print(f"F-Statistic: {res.statistic}, p-value: {res.pvalue}")
```

In case the sphericity assumption was not met, you can choose to apply the Greenhouse-Geisser, Huynh-Feldt, or Lower Bound correction via the `correction` parameter as shown below:

```python
res_greenhouse_geisser_correction = f_repeated(
    sample1, sample2, sample3, correction="greenhouse"
)

res_huynh_feldt_correction = f_repeated(sample1, sample2, sample3, correction="huynh")

res_lower_bound_correction = f_repeated(sample1, sample2, sample3, correction="lower")
```

Note that the `correction` parameter is optional and set to `None` by default, implying how the sphericity assumption was met.

If you found your results were significant from the `f_repeated` function, you can run a pairwise paired t-test with p-value correction using the `pairwise_ttest` as shown below:

```python
from anovapy.oneway import pairwise_ttest

res = pairwise_ttest(sample1, sample2, sample3, adjust="bonferroni")

res_holm = pairwise_ttest(
    sample1, sample2, sample3, adjust="holm"
)  # or use holm adjustment

print(f"Statistics: {res.statistic}")
print(f"p-values: {res.pvalue}")
```

Note that `statistic` and `pvalue` in this case is a list of values, where each index represents a comparison. For example, `res.statistic[0]` represents the comparison of `sample1` and `sample2`, `res.statistic[1]` represents the comparison of `sample1` and `sample3` and so on.

## Permutation One Way ANOVA

Another common test run when assumptions of One Way ANOVA are not met is the Permutation One Way ANOVA, which you can run with:

```python
from anovapy.oneway import f_permutation

# let's reuse sample1, sample2, and sample3 from earlier

res = f_permutation(sample1, sample2, sample3)

print(f"F-Statistic: {res.statistic}, p-value: {res.pvalue}")
```

The `f_permutation` function also has some parameters you can mess around with, more specifically the `permutations` parameter which lets you set the number of permutations to run and the `random_state` parameter so you can set a custom seed for reproducible results.

A word on the `permutations` parameter. This controls the number of random label shuffles used to build the null distribution of the F-statistic. The smallest p-value that can be reported is `1 / (permutations + 1)`, so choose a value whose floor is well below your significance threshold (this is what is recommended by the way when setting the number of permutations):

- **999** (default) -- minimum $p = 0.001$, sufficient for testing at $\alpha = 0.05$.
- **4999** -- minimum $p \approx 0.0002$, sufficient for testing at $\alpha = 0.01$.

Increasing the number of permutations beyond these values improves the precision of the p-value (especially the third decimal place) but has diminishing returns. If the number of permutations exceeds the total number of unique label arrangements for your data, the test becomes exact.

The examples below demonstrate their usage accordingly:

```python
res2 = f_permutation(sample1, sample2, sample3, permutations=4999)

res3 = f_permutation(sample1, sample2, sample3, permutations=99, random_state=42)
```

## One Way ANCOVA

AnovaPy also let's you run a One Way Analysis of Covariance (ANCOVA) via the `f_ancova` function. You can pass in one or more covariates very easily:

```python
from anovapy.oneway import f_ancova

# let's generate some fresh data
labels = np.repeat(["A", "B", "C"], 10)
dependent = np.random.normal(size=30)
covariate = np.random.normal(size=30)

res = f_ancova(labels, dependent, covariate)

print(f"F-Statistic: {res.statistic}, p-value: {res.pvalue}")

for i in range(len(res.covariate_stats)):
    print(
        f"Covariate F-Statistic: {res.covariate_stats[i]}, p-value: {res.covariate_pvalues[i]}"
    )

# or pass in more covariates

covariate2 = np.random.normal(size=30)
covariate3 = np.random.normal(size=30)

res_more_covariate_example = f_ancova(
    labels, dependent, covariate, covariate2, covariate3
)

print(
    f"F-Statistic: {res_more_covariate_example.statistic}, p-value: {res_more_covariate_example.pvalue}"
)

for i in range(len(res_more_covariate_example.covariate_stats)):
    print(
        f"Covariate F-Statistic: {res_more_covariate_example.covariate_stats[i]}, p-value: {res_more_covariate_example.covariate_pvalues[i]}"
    )
```

Note that in `covariate_stats` and the associated `covariate_pvalues`, each item's index there is proportional to how the covariate was passed in. For example, in `res_more_covariate_example.covariate_stats`, F-statistic at index 0 refers to the F-statistic (and the accompanying p-value) for `covariate`, the F-statistic at index 1 refers to the F-statistic for `covariate2` and so on.

## ART One Way ANOVA

AnovaPy also implements the Aligned Rank Transform (ART) ANOVA for One Way designs.

*Note: ART is primarily designed for factorial (Two Way and N-Way) ANOVA models. For One Way designs, consider Welch's ANOVA (`scipy.stats.f_oneway` with `equal_var=False`), Brown-Forsythe ANOVA (`anovapy.oneway.brown`), the Kruskal-Wallis test (`scipy.stats.kruskal`), or a Permutation One Way ANOVA (`anovapy.oneway.f_permutation`) depending on your data and assumptions. This function is included for API consistency across factorial/One Way workflows and parity with ARTool.*

This can easily be run as follows:

```python
from anovapy.oneway import art

sample1 = np.random.normal(size=100)
sample2 = np.random.normal(size=100)
sample3 = np.random.normal(size=100)

res = art(sample1, sample2, sample3)

print(f"F-Statistic: {res.statistic}, p-value: {res.pvalue}")
```

You can also run the post-hoc, ART-Contrast, as follows:

```python
from anovapy.oneway import artc

res = artc(sample1, sample2, sample3)

for i in range(len(res.statistic)):
    print(
        f"For comparison: {res.comparisons[i]}, t-Statistic: {res.statistic[i]} and p-value: {res.pvalue[i]}"
    )
```

Once again, like with other post-hoc tests implemented in AnovaPy, you can mess around with the `adjust` parameter to adjust the p-value in any way you like:

```python
res_holm = artc(sample1, sample2, sample3, adjust="holm")
```

If not explicitly set, `adjust` will default to `'bonferroni'`.