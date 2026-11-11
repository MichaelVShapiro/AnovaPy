"""
Multivariate Analysis of Variance (MANOVA) module for AnovaPy.

Provides lightweight, NumPy-native implementations for multivariate test statistics, assumption checks, and post-hoc analyses.
"""

from .core import f_manova, transform_and_run_manova, transform_dependent
from .kruskal import multivariate_kruskal, univariate_kruskal
from .nonparametric import nonparametric_manova
from .permanova import permanova

__all__ = [
    "f_manova",
    "multivariate_kruskal",
    "nonparametric_manova",
    "permanova",
    "transform_and_run_manova",
    "transform_dependent",
    "univariate_kruskal",
]
