# AnovaPy Documentation

Welcome to AnovaPy's documentation! AnovaPy is generally a small library. It is designed to be lightweight and get statistical computations done fast.

The #1 goal of AnovaPy is to minimize the learning curve by providing APIs that take plain **NumPy arrays** as inputs. Function signatures mimic those of **SciPy** (`scipy.stats`). If you know how to plug an array into a SciPy function like `scipy.stats.pearsonr`, you already know how to use AnovaPy -- it is that intuitive!

Overall, all documentation in this folder is therefore defined strictly as follows:

- `OneWayTests.md` -> How to run One Way ANOVA including other "alternative" tests (when assumptions are not met)
- `TwoWayTests.md` -> How to run Two Way ANOVA including other "alternative" tests.
- `Multivariate.md` -> How to run MANOVA including other "alternative" tests.
- `MultivariateNormality.md` -> How to check if multivariate data is normally distributed. This is a critical assumption of MANOVA.
- `Sphericity.md` -> How to check if data meets the sphericity assumption needed for tests like Repeated Measures ANOVA.