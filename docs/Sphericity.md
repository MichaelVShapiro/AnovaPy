# Checking Sphericity

One test AnovaPy lets you run is the Repeated Measures ANOVA test. However, before you run that test you must check if your data meets the sphericity assumption. AnovaPy implements a few tests to check this.

## Table of Contents

- [Mauchly Test](#mauchly-test)
- [John-Nagao-Sugiura Test](#john-nagao-sugiura-test)

## Mauchly Test

You can run the Mauchly using the `mauchly` function as shown below:

```python
from anovapy.sphericity import mauchly
import numpy as np

# make up some data
samples_a = np.random.normal(size=100)
samples_b = np.random.normal(size=100)
samples_c = np.random.normal(size=100)

res = mauchly(samples_a, samples_b, samples_c)

print(f"W-Statistic: {res.statistic}, p-value: {res.pvalue}")
```

It is as simple as that to run the test!

## John-Nagao-Sugiura Test

AnovaPy also supports the ability to run the John-Nagao-Sugiura test, and this can be achieved via the `anovapy.sphericity.john` function as shown below:

```python
from anovapy.sphericity import john

res = john(samples_a, samples_b, samples_c)

print(f"T-Statistic: {res.statistic}, p-value: {res.pvalue}")
```