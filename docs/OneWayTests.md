# One Way Tests

Generally speaking, SciPy already implements One Way ANOVA via the `f_oneway` function (see `scipy.stats.f_oneway`). It also implements "alternative" tests (in case assumptions aren't met).

For this reason, there aren't many tests AnovaPy implements at the moment regarding One Way ANOVA and other related non-parametric tests.

The below table of contents summarizes all ANOVA-related tests supported by AnovaPy.

## Table of Contents

- [Dunn's Post-Hoc](#dunns-post-hoc)
- [Brown-Forsythe ANOVA Test](#brown-forsythe-anova-test)
- [One Way Repeated Measures ANOVA](#one-way-repeated-measures-anova)

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