# Contributing

First off, thank you for considering to contribute to AnovaPy! My vision for AnovaPy is to make a super lightweight library that implements ANOVA and ANOVA-related functions all under a permissive license.

## Getting Started

Make sure you fork this repo and clone it on your local machine. Install AnovaPy locally with `pip install -e .[test]`. This will install all dependencies needed for AnovaPy to work and all dependencies needed to run test cases.

Note that PyTest is used to run test cases. If you are not familiar with how it works, check out a sample test case file (e.g. `tests/test_normal.py`) to see how to make test cases. You can run test cases simply with:

```bash
cd tests
pytest test_normal.py # or a different file!
```

## A Few Rules

This section will just briefly cover some general rules when contributing. Subsequent sections cover specific guidelines.

- Keep all contributions relevant to working with ANOVA. This library wouldn't be called `AnovaPy` otherwise! Note that "relevant" here is vague. It could mean implementing a non-parametric alternative to One-Way, Two-Way, or Multivariate ANOVA. It could mean implementing a post-hoc test or tests that check assumptions. Just as long as it is related to ANOVA, it can be implemented here.
- **Do not** make use of any GPL/copyleft code. AnovaPy has been created under the MIT license, a permissive license, and any code used under a different license should be compatible with this one. For example, code in R is usually licensed under GPL, hence trying to transcompile it to use in AnovaPy is strictly forbidden. Feel free to use R (or any other tool) for testing purposes but not for development.
- Understand the license your code will be under. AnovaPy is licensed under MIT. By submitting your code to AnovaPy, you acknowledge that your code will also be licensed under MIT.
- It's okay to use AI! If it helps optimize code or even just understand something in research papers better, that is fine! The only policy here with using AI is to *always check the output.* For instance, if the AI writes and/or references code written in another library under GPL or makes suboptimal solutions, those should not be included in AnovaPy.
- Always cite a reliable source to use. Whether this is a research paper, or a website containing some formulas, these are important to help make the logic in AnovaPy.

### Making Issues

Before making an issue, please check if a similar issue is already opened.

When making an issue, please assign it an appropriate label (e.g. "Bug"). For any feature requests, please note down its relevance to AnovaPy and its mission. Any features or statistical tests not relevant may be rejected.

### Pull Requests

In addition to the general rules stated previously, please make sure your code conforms to the following rules in AnovaPy before opening a pull request:

- All functions are documented clearly. All functions implementing new tests should have at least one primary source cited (all DOI papers should be cited in APA format). Any secondary sources used should be cited in MLA8 format.
- All functions should have a place under the `docs/` folder and be documented there as needed so any developer understands how to use it.
- Function signatures should mimic SciPy where applicable (accepting NumPy arrays, supporting parameters like `nan_policy`, etc.) to minimize the learning curve.
- All existing test cases must pass, and new features must include comprehensive PyTest test cases.
- AnovaPy uses [Ruff](https://github.com/astral-sh/ruff) for code formatting and linting. Before opening a pull request, make sure you use Ruff for linting and formatting as needed!

## Wrapping Up

Once again, thank you for your contributions! I hope AnovaPy will be super useful and be something generally easy for one to work with.