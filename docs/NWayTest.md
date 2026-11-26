# N-Way ANOVA Tests

AnovaPy implements a pretty easy way to run Three Way ANOVA, Four Way ANOVA, and beyond. This document describes how.

## Table of Contents

- [N-Way ANOVA](#n-way-anova)
- [ART ANOVA](#art-anova)

## N-Way ANOVA

So a bit of a recap on how N-Way designs work. The "N" there really refers to how many factors (independent variables) you have in your design. For example, in a *Two Way* ANOVA you only have *two* factors and in a *Three Way* ANOVA you have *three* factors and so on. Regardless of the number of factors you have, you will always have *one* dependent variable. The simplest way to use the API for N-Way designs is as follows therefore:

```python
from anovapy.nway import f_nway
import numpy as np

# we will take this example from: https://www.statology.org/three-way-anova-python/

program = np.repeat([1, 2], 20)
gender = np.tile(np.repeat(["M", "F"], 10), 2)
division = np.tile(np.repeat([1, 2], 5), 4)
height = np.array(
    [
        7,
        7,
        8,
        8,
        7,
        6,
        6,
        5,
        6,
        5,
        5,
        5,
        4,
        5,
        4,
        3,
        3,
        4,
        3,
        3,
        6,
        6,
        5,
        4,
        5,
        4,
        5,
        4,
        4,
        3,
        2,
        2,
        1,
        4,
        4,
        2,
        1,
        1,
        2,
        1,
    ]
)

# in this case the three factors are: program, gender, and division with height being our dependent variable

res = f_nway(program, gender, division, dependent=height)

for i in range(len(res.statistic)):
    print(f"Comparising effect: {res.effect[i]}")
    print(f"F-Statistic: {res.statistic[i]}")
    print(f"p-value: {res.pvalue[i]}")
    print(f"Degrees of Freedom: {res.df[i]}")

print(f"Degrees of Freedom Error: {res.df_error}")
```

It doesn't really matter how many independent variables you add into `f_nway`. Here is another example (note that running this won't actually work -- it is simply meant to demonstrate how the N-Way design works and how to pass it into AnovaPy):

```python
factor4 = np.repeat(["A", "B"], 20)

res2 = f_nway(program, gender, division, factor4, dependent=height)

factor5 = np.repeat(["C", "D"], 20)

res3 = f_nway(program, gender, division, factor4, factor5, dependent=height)

factor6 = np.repeat(["E", "F"], 20)

res4 = f_nway(program, gender, division, factor4, factor5, factor6, dependent=height)
```

So that really is the pattern for `f_nway` -- you can add as many factors as you like before setting the `dependent` parameter.

The return result is also pretty important to understand.

Note that `res.effect`, `res.statistic`, `res.pvalue`, and `res.df` are all NumPy arrays of the same length where index *i* of `res.effect` corresponds to index *i* of `res.statistic` and `res.pvalue` respectively.

While it may make more sense what `res.statistic` and `res.pvalue` represet, it may be less clear what `res.effect` has. Generally speaking, the array contains a whole bunch of tuples of different lengths! Each tuple can have one or more numbers representing indecies of each item passed into the first parameter of `f_nway`.

For example, if we examine the response of `res.effect[0]` we will find how it has the tuple `(0,)` implying how a single Factor (more specifically the first item passed into `f_nway` which in this case is `program`) is being analyzed for that effect. Later we can find a tuple that has the value `(0, 1)` (or `(0, 2)` or `(1, 2)`) meaning we are analzying the *interaction* betwene *program* (first item) and *gender* (second item). Since `res` was generated from a Three Way design, a tuple analyzing `program:gender:division` will be represented by the tuple `(0, 1, 2)` respectively.

The final most important thing to note about this function is that it should **NOT** be used for handling Two Way designs (i.e. Two Way ANOVA). Use `anovapy.twoway.f_twoway` which is optimized for handling Two Way designs.

AnovaPy also implements the Tukey HSD post-hoc test for N-Way designs via the `anovapy.nway.tukey_hsd` function. In an N-Way design, you rarely want to run post-hoc comparisons on every main effect and interaction at once (especially considering how some may not have shown to be significant when running the N-Way ANOVA). `anovapy.nway.tukey_hsd` lets you target a specific effect using the `term` parameter.

Pass your independent factors positionally, and pass a tuple of their 0-based indices to `term` to specify which main effect or interaction to test:

```python
from anovapy.nway import tukey_hsd

# we will reuse the same factors as before
# as a reminder it is: program (0), gender (1), and division (2)
# lets test the main effect of 'program' (index 0)
res_main = tukey_hsd(program, gender, division, dependent=height, term=(0,))

# now lets test the 2-way interaction between 'gender' (1) and 'division' (2)
res_2way = tukey_hsd(program, gender, division, dependent=height, term=(1, 2))

# now lets test the 2-way interaction between 'program' (0) and 'division' (2)
res_alt_2way = tukey_hsd(program, gender, division, dependent=height, term=(0, 2))

# let test the 3-way interaction across all factors
res_3way = tukey_hsd(program, gender, division, dependent=height, term=(0, 1, 2))

# finally lets see the results
for comp, q_stat, p_val in zip(
    res_main.comparisons, res_main.statistic, res_main.pvalue
):
    print(f"For comparison {comp}: Q-Statistic = {q_stat:.3f}, p-value = {p_val:.4f}")
```

A word on the return result for `anovapy.nway.tukey_hsd`. The function returns a dataclass where elements at the same index `i` across `comparisons`, `statistic`, and `pvalue` always correspond to the same comparison.

Rather than joining factor labels into a single string (like `"A1:B1"`), each entry in `res.comparisons` is structured as `(group_a, group_b)`, where both `group_a` and `group_b` are tuples of factor levels matching the order of indices in `term`:

- **Main effect** (`term=(0,)`): `(('ProgramA',), ('ProgramB',))`
- **2-Way Interaction** (`term=(1, 2)`): `(('Female', 'East'), ('Male', 'East'))`
- **3-Way Interaction** (`term=(0, 1, 2)`): `(('ProgramA', 'Female', 'East'), ('ProgramB', 'Female', 'East'))`

Because the levels stay separated in tuples, you can format or unpack them however you want:

```python
for (g1, g2), q_stat, p_val in zip(res.comparisons, res.statistic, res.pvalue):
    left = " : ".join(map(str, g1))
    right = " : ".join(map(str, g2))
    print(f"{left} vs {right} -> Q-Statistic={q_stat:.3f}, p-value={p_val:.4f}")
```

## ART ANOVA

In case assumptions for N-Way ANOVA are not met, AnovaPy implements the Aligned Rank Transform (ART) ANOVA. This can easily be achieved as follows:

```python
from anovapy.nway import art

res_art = art(program, gender, division, dependent=height)
```

Note that the return result of `art` is identical to that of `f_nway`.

Another important thing to note here is that just like with `f_nway`, you should **NOT** use `art` for handling Two Way designs (i.e. Two Way ANOVA). Use `anovapy.twoway.art` which is optimized for handling Two Way designs.