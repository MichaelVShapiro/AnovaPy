# Two Way Tests

AnovaPy implements the Two Way ANOVA test and a variety of other "alternative" tests (in case assumptions are not met). The below describes each.

## Table of Contents

- [Two Way ANOVA](#two-way-anova)
- [Scheirer-Ray-Hare Test](#scheirer-ray-hare-test)
- [ART ANOVA](#art-anova)
- [Transforming Dependent Variable](#transforming-dependent-variable)

## Two Way ANOVA

To run a standard Two Way ANOVA, you can use the `f_twoway` (meant to have a similar naming convention to SciPy's `f_oneway` function) as shown below:

```python
from anovapy.twoway import f_twoway
import numpy as np

factor_a = np.random.normal(size=60)
factor_b = np.random.normal(size=60)
dependent = np.random.normal(size=60)

twoway_res = f_twoway(factor_a=factor_a, factor_b=factor_b, dependent=dependent)

print(f"Statistic A result: {twoway_res.statistic_a}")
print(f"p-value A result: {twoway_res.pvalue_a}")
```

To run Tukey's post hoc test (assuming the Two Way ANOVA result proves significant of course), you can call `tukey_hsd` as shown below:

```python
from anovapy.twoway import tukey_hsd

tukey_res = tukey_hsd(factor_a=factor_a, factor_b=factor_b, dependent=dependent)

for i in range(len(tukey_res.comparisons)):
    label1, label2 = tukey_res.comparisons[i]
    print(
        f"For comparison {label1}:{label2}: Statistic: {tukey_res.statistic[i]}, p-value: {tukey_res.pvalue[i]}"
    )
```

Overall, the function is designed to mirror the return result of SciPy's `scipy.stats.tukey_hsd` as closely as possible. Hence `tukey_res.statistic` represents the computed statistic for each comparison and `tukey_res.pvalue` represents the computed p-value for each comparison. To view each comparison, see `tukey_res.comparisons`, which stores a tuple of the two labels being compared. When analyzing just Factor A or Factor B, the tuple contains only two values. When analyzing the interaction, the tuple contains two values that are tuples containing two more values of each item in the interaction (see example below).

It is also important to note how you can pass in which effect or interaction to analyze in the `tukey_hsd` function using the `term` parameter as shown below:

```python
tukey_res_a = tukey_hsd(
    factor_a=factor_a, factor_b=factor_b, dependent=dependent, term="factor_a"
)  # analyzes effect on factor a

tukey_res_b = tukey_hsd(
    factor_a=factor_a, factor_b=factor_b, dependent=dependent, term="factor_b"
)  # analyzes effect on factor b

tukey_res_interaction = tukey_hsd(
    factor_a=factor_a, factor_b=factor_b, dependent=dependent, term="interaction"
)  # analyzes interaction

# note that for interaction, the comparisons are stored in format: ((label1, label2), (label3, label4))
# which can essentially be represented as analyzing: label1:label2-label3:label4
# the example below demonstrates this
for i in range(len(tukey_res_interaction.comparisons)):
    label1 = post_hoc_interaction.comparisons[0][0]
    label2 = post_hoc_interaction.comparisons[0][1]
    label3 = post_hoc_interaction.comparisons[1][0]
    label4 = post_hoc_interaction.comparisons[1][1]
    print(f"Comparing: {label1}:{label2} - {label3}:{label4}")
    print(
        f"Statistic: {post_hoc_interaction.statistic[i]}, p-value: {post_hoc_interaction.pvalue[i]}"
    )
```

If the `term` parameter is not specified, it will default to `factor_a`.

*Note: unlike SciPy stats `tukey_hsd` (see `scipy.stats.tukey_hsd`) which is designed to handle the post-hoc test for a One Way ANOVA and features the `confidence_inveral` function as a part of its result, as of v0.1 in AnovaPy, there is no such `confidence_interval` function yet, though there are plans to implement this in the future!*

## Scheirer-Ray-Hare Test

AnovaPy also supports running the Scheirer-Ray-Hare test in case an assumption is not met for Two Way ANOVA. You can use the `scheirer` function which has a similar signature to `f_twoway`:

```python
from anovapy.twoway import scheirer

# we are reusing the factor_a, factor_b, and dependent already made in the previous example
scheirer_res = scheirer(factor_a=factor_a, factor_b=factor_b, dependent=dependent)

print(f"Statisitc: {scheirer_res.statistic}, p-value: {scheirer_res.pvalue}")
```

## ART ANOVA

Another popular test used when assumptions aren't met for Two Way ANOVA is the Aligned Rank Transform (ART) Two Way ANOVA. You can run this test using the `art` function. The `artc` function can be used to run the post-hoc test (this follows the ART-C framework) as shown below:

```python
from anovapy.twoway import art, artc

art_res = art(factor_a=factor_a, factor_b=factor_b, dependent=dependent)

print(f"Factor A stat and p-value: {art_res.statistic_a}, {art_res.pvalue_a}")
print(f"Factor B stat and p-value: {art_res.statistic_b}, {art_res.pvalue_b}")
print(
    f"Interaction stat and p-value: {art_res.statistic_interaction}, {art_res.pvalue_interaction}"
)

# now run post-hoc
post_hoc_res = artc(factor_a=factor_a, factor_b=factor_b, dependent=dependent)

for i in range(len(post_hoc_res.comparisons)):
    comp1, comp2 = post_hoc_res.comparisons[i]
    print(
        f"At comparison: {comp1}-{comp2}, there is statistic: {post_hoc_res.statistic[i]} and adjusted p-value of {post_hoc_res.pvalue[i]}"
    )
```

Once again, we can mess around with the `term` parameter to select the effect/factor to test against and even play around with the `adjust` parameter to select the p-value adjustment method:

```python
post_hoc_a = artc(
    factor_a=factor_a, factor_b=factor_b, dependent=dependent, term="factor_a"
)
post_hoc_b = artc(
    factor_a=factor_a, factor_b=factor_b, dependent=dependent, term="factor_b"
)
post_hoc_interaction = artc(
    factor_a=factor_a, factor_b=factor_b, dependent=dependent, term="interaction"
)

# note: since term='interaction' analyzes two cell means, the comparison is typically something along the lines of A1:B1-A1:B2
# as opposed to term='factor_a' or term='factor_b' therefore, AnovaPy returns a tuple that has two separate tuple values just like the tukey_hsd function
# so you can check the values with:
for i in range(len(post_hoc_interaction.comparisons)):
    label1 = post_hoc_interaction.comparisons[0][0]
    label2 = post_hoc_interaction.comparisons[0][1]
    label3 = post_hoc_interaction.comparisons[1][0]
    label4 = post_hoc_interaction.comparisons[1][1]
    print(f"Comparing: {label1}:{label2} - {label3}:{label4}")

# p-value adjustment method
post_hoc_holm = artc(
    factor_a=factor_a, factor_b=factor_b, dependent=dependent, adjust="holm"
)
post_hoc_bonferroni = artc(
    factor_a=factor_a, factor_b=factor_b, dependent=dependent, adjust="bonferroni"
)
post_hoc_no_adjust = artc(
    factor_a=factor_a, factor_b=factor_b, dependent=dependent, adjust="none"
)
```

⚠️ Warning: the `artc` function also has the `interaction` parameter, set to `False` by default which tests for differences of differences. This is currently NOT implemented as of v0.1. Therefore, running the below will result in an error:

```python
res_diff_of_diff = artc(
    factor_a=factor_a,
    factor_b=factor_b,
    dependent=dependent,
    term="interaction",
    interaction=True,
)  # will throw an error since interaction=True has not been implemented yet
```

**Why is `interaction=True` not implemented yet?** One of the core design pillars of AnovaPy is to remain **lightweight,** dependening exclusively on NumPy and SciPy without pulling in heavy modeling stacks. There was some difficulty developing `interaction=True` for v0.1. Taken from the official academic paper (see \[1\] in the doc string for `artc`), when `interaction=True` (more specifically for ARTool's `art.con` function) the following is executed in R:

```r
# Fit ART model to raw response
m.art = art(Y ~ A*B*C + (1|S), data=df)

# Conduct A x B contrasts on ART model
contrast(emmeans(artlm(m.art, "A:B"), ~ A:B), method="pairwise", adjust="holm")
```

In other words, while the formula and logic is written in the paper, there was no clear way to implement it for AnovaPy v0.1. Since AnovaPy is licensed under MIT and `emmeans`, `constrast` (both part of the `emmeans` R library), and `art.con` are all under GPL (copyleft), code cannot just be transcompiled that easily.

*Note: community contributions are welcome!! If you are a researcher or developer who understands linear contrast algebra and wants to help implement `interaction=True` for the `artc` function, pull requests are warmly welcomed!*

> **Note on Numerical Precision & R Compatibility**
> 
> In rare edge cases, `art()` outputs may differ slightly from R's `ARTool`. This is expected and stems from underlying differences in floating-point handling. While R/ARTool effectively rounds or formats intermediate values around the 8th decimal place, AnovaPy uses NumPy's `float64` double precision.
> 
> In extreme boundary conditions -- such as the edge case in `test_art_simple` (see `tests/test_twoway.py::TestTwoWayAnova`) where values diverge only at the 15th decimal place -- this subtle difference affects rank tie-breaking. Consequently, AnovaPy's implementation retains higher mathematical fidelity, offering greater numerical reliability in these rare instances.

## Transforming Dependent Variable

Another popular method before running a Two Way ANOVA (in case assumptions aren't met) is to transform the dependent variable. AnovaPy implements this natively.

```python
from anovapy.twoway import transform_dependent, transform_and_run_twoway

updated_dependent = transform_dependent(dependent=dependent)

# or provide an exact method of transformation
updated_dependent_log = transform_dependent(dependent=dependent, method="log")
updated_dependent_square_root = transform_dependent(
    dependent=dependent, method="square_root"
)
updated_dependent_reciprocal = transform_dependent(
    dependent=dependent, method="reciprocal"
)
updated_dependent_box_cox = transform_dependent(dependent=dependent, method="box_cox")

# then run two way after the transformation
twoway_res = f_twoway(
    factor_a=factor_a, factor_b=factor_b, dependent=updated_dependent
)  # or change dependent with another transformation done

# or apply the transformation and run two way automatically
twoway_res_auto = transform_and_run_twoway(
    factor_a=factor_a, factor_b=factor_b, dependent=dependent, method="log"
)

# twoway_res_auto and twoway_res are identical
```

Note that when `method` is left out, it will default to `log`.

A word on `transform_and_run_twoway` and `transform_dependent`. Typically, after transforming the dependent variable, you would run a standard Two Way ANOVA test. You will notice in the code above how after calling `transform_dependent` we ran `f_twoway` automatically. `transform_and_run_twoway`, therefore, is literally just calling `transform_dependent` and then `f_twoway` right away.