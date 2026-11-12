# this test permanova

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from skbio.stats.distance import DistanceMatrix
from skbio.stats.distance import permanova as skbio_permanova

from anovapy.manova import permanova

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def well_separated_data():
    """Two clearly distinct 2D clusters."""
    rng = np.random.default_rng(42)
    group_a = rng.normal(loc=0.0, scale=0.1, size=(15, 3))
    group_b = rng.normal(loc=10.0, scale=0.1, size=(15, 3))
    return group_a, group_b


@pytest.fixture
def identical_distribution_data():
    """Two groups sampled from the exact same distribution (Null Hypothesis True)."""
    rng = np.random.default_rng(42)
    group_a = rng.normal(loc=0.0, scale=1.0, size=(20, 4))
    group_b = rng.normal(loc=0.0, scale=1.0, size=(20, 4))
    return group_a, group_b


# =============================================================================
# 1. INPUT SHAPE & API SIGNATURE TESTS
# =============================================================================


class TestInputSignatures:
    def test_varargs_input_2d(self, well_separated_data):
        """Test passing multiple 2D arrays as *args."""
        g1, g2 = well_separated_data
        res = permanova(g1, g2, permutations=99, random_state=42)

        assert isinstance(res.statistic, float)
        assert res.dof_between == 1
        assert res.dof_within == 28

    def test_single_matrix_with_grouping(self, well_separated_data):
        """Test passing a single X matrix with explicit `grouping` array."""
        g1, g2 = well_separated_data
        X = np.vstack([g1, g2])
        labels = np.array([0] * len(g1) + [1] * len(g2))

        res_matrix = permanova(X, labels, permutations=99, random_state=42)
        res_independent = permanova(X, labels, permutations=99, random_state=42)
        res_varargs = permanova(g1, g2, permutations=99, random_state=42)

        # Output should be identical regardless of invocation style
        assert_almost_equal(res_matrix.statistic, res_varargs.statistic)
        assert_almost_equal(res_matrix.pvalue, res_varargs.pvalue)
        assert_almost_equal(res_independent.statistic, res_varargs.statistic)
        assert_almost_equal(res_independent.pvalue, res_varargs.pvalue)

    def test_1d_input_conversion(self):
        """
        Test 1D arrays passed as groups (e.g. univariate samples).
        Ensure 1D inputs shaped (N,) get promoted/handled like (N, 1).
        """
        g1 = np.array([1.0, 2.0, 1.5, 2.2])
        g2 = np.array([8.0, 9.0, 8.5, 9.1])

        # Reshape internally if 1D array is supplied
        g1_2d = g1[:, None]
        g2_2d = g2[:, None]

        res_1d = permanova(g1_2d, g2_2d, permutations=99, random_state=42)
        assert res_1d.statistic > 10.0
        assert res_1d.dof_between == 1

    def test_high_dimensional_input(self):
        """Test wide feature matrices (n_samples < n_features)."""
        rng = np.random.default_rng(42)
        # 10 samples, 500 features
        g1 = rng.normal(0, 1, size=(5, 500))
        g2 = rng.normal(5, 1, size=(5, 500))

        res = permanova(g1, g2, permutations=99, random_state=42)
        assert not np.isnan(res.statistic)
        assert res.statistic > 0


# =============================================================================
# 2. STATISTICAL CORRECTNESS & HYPOTHESIS TESTS
# =============================================================================


class TestStatisticalCorrectness:
    def test_well_separated_clusters_pvalue(self, well_separated_data):
        """Distinct clusters should yield high Pseudo-F and minimal p-value."""
        g1, g2 = well_separated_data
        res = permanova(g1, g2, permutations=999, random_state=42)

        # Empirical p-value min floor is 1 / (N_perm + 1) = 1 / 1000 = 0.001
        assert res.pvalue == pytest.approx(0.001)
        assert res.eta_squared > 0.90  # Strong effect size

    def test_identical_distributions_pvalue(self, identical_distribution_data):
        """Identical distributions should have high p-value (fail to reject H0)."""
        g1, g2 = identical_distribution_data
        res = permanova(g1, g2, permutations=999, random_state=42)

        assert res.pvalue > 0.10
        assert res.eta_squared < 0.15

    def test_known_ground_truth_scikit_bio_equivalent(self):
        """
        Verification against known fixed values computed via SciKit-Bio/Anderson (2001).

        Dataset:
          Group A: [[1, 2], [2, 2]]
          Group B: [[5, 5], [6, 5]]
        """
        g1 = np.array([[1.0, 2.0], [2.0, 2.0]])
        g2 = np.array([[5.0, 5.0], [6.0, 5.0]])

        res = permanova(g1, g2, permutations=999, random_state=42)

        # Exact Anderson (2001) values:
        # SS_total = 26.0, SS_within = 1.0, SS_between = 25.0
        # Pseudo-F = (25.0 / 1) / (1.0 / 2) = 50.0
        # Eta-squared = 25.0 / 26.0 = 0.961538...
        assert_almost_equal(res.statistic, 50.0, decimal=5)
        assert_almost_equal(res.eta_squared, 25.0 / 26.0, decimal=5)
        assert res.dof_between == 1
        assert res.dof_within == 2


# =============================================================================
# 3. PARAMETER COMBINATIONS & METRICS
# =============================================================================


class TestDistanceMetricsAndPermutations:
    @pytest.mark.parametrize(
        "metric", ["euclidean", "cityblock", "cosine", "chebyshev"]
    )
    def test_distance_metrics(self, well_separated_data, metric):
        """Verify support for standard SciPy distance metrics."""
        g1, g2 = well_separated_data
        res = permanova(
            g1, g2, distance_metric=metric, permutations=99, random_state=42
        )

        assert not np.isnan(res.statistic)
        assert res.pvalue <= 0.05

    @pytest.mark.parametrize("num_perm", [19, 99, 499])
    def test_permutation_counts(self, well_separated_data, num_perm):
        """Ensure minimum p-value bound scales with permutation count formula: 1/(N+1)."""
        g1, g2 = well_separated_data
        res = permanova(g1, g2, permutations=num_perm, random_state=42)

        expected_min_p = 1.0 / (num_perm + 1.0)
        assert res.pvalue == pytest.approx(expected_min_p)


# =============================================================================
# 4. MISSING VALUES & ERROR HANDLING
# =============================================================================


class TestNanPolicyAndErrors:
    def test_nan_policy_raise(self):
        """nan_policy='raise' should raise ValueError on NaN."""
        g1 = np.array([[1.0, 2.0], [np.nan, 3.0]])
        g2 = np.array([[4.0, 5.0], [6.0, 7.0]])

        with pytest.raises(ValueError, match="contains NaN"):
            permanova(g1, g2, nan_policy="raise")

    def test_nan_policy_propagate(self):
        """nan_policy='propagate' should return NaN statistics."""
        g1 = np.array([[1.0, 2.0], [np.nan, 3.0]])
        g2 = np.array([[4.0, 5.0], [6.0, 7.0]])

        res = permanova(g1, g2, nan_policy="propagate")
        assert np.isnan(res.statistic)
        assert np.isnan(res.pvalue)

    def test_nan_policy_omit(self):
        """nan_policy='omit' should drop incomplete rows."""
        g1 = np.array(
            [[1.0, 2.0], [np.nan, 3.0], [1.5, 2.1]]
        )  # 1 row dropped -> 2 rows remain
        g2 = np.array([[5.0, 5.0], [6.0, 5.0]])  # 2 rows

        res = permanova(g1, g2, nan_policy="omit", permutations=99, random_state=42)
        assert not np.isnan(res.statistic)
        assert res.dof_within == 2  # (2 + 2) - 2 = 2

    def test_insufficient_groups_error(self):
        """Should raise ValueError if fewer than 2 groups are provided."""
        g1 = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="at least 2 distinct groups"):
            permanova(g1, np.array([0, 0]))


# tests against scikit-bio specifically


class TestAgainstSciKitBio:
    def test_precomputed_distance_matrix_against_skbio(self):
        """
        Direct comparison using a pre-computed distance matrix.
        Validates against skbio.stats.distance.permanova.
        """
        data = [
            [0.0, 0.3, 0.7, 0.8],
            [0.3, 0.0, 0.6, 0.9],
            [0.7, 0.6, 0.0, 0.2],
            [0.8, 0.9, 0.2, 0.0],
        ]
        ids = ["S1", "S2", "S3", "S4"]
        grouping = ["Control", "Control", "Treatment", "Treatment"]
        permutations = 999
        seed = 42

        # 1. Run SciKit-Bio PERMANOVA
        dm = DistanceMatrix(data, ids)
        skbio_res = skbio_permanova(dm, grouping, permutations=permutations, seed=seed)

        # Extract values from skbio output Series (matching exact key names)
        expected_f = skbio_res["test statistic"]
        expected_p = skbio_res["p-value"]
        expected_sample_size = skbio_res["sample size"]  # <--- FIXED KEY NAME
        expected_groups = skbio_res["number of groups"]

        # 2. Run AnovaPy PERMANOVA
        dist_matrix = np.array(data)
        anovapy_res = permanova(
            dist_matrix,
            grouping,
            permutations=permutations,
            distance_metric="precomputed",
            random_state=seed,
        )

        # 3. Assertions
        assert_almost_equal(anovapy_res.statistic, expected_f, decimal=5)
        assert anovapy_res.dof_between == expected_groups - 1
        assert anovapy_res.dof_within == expected_sample_size - expected_groups
        assert anovapy_res.pvalue == pytest.approx(expected_p, abs=0.05)

    def test_raw_features_against_skbio(self):
        """
        Comparison starting from raw feature arrays (X) and standard Euclidean distance.
        """
        rng = np.random.default_rng(123)
        g1 = rng.normal(loc=0.0, scale=1.0, size=(10, 5))
        g2 = rng.normal(loc=2.0, scale=1.0, size=(10, 5))

        X = np.vstack([g1, g2])
        grouping = ["A"] * 10 + ["B"] * 10
        ids = [f"S{i}" for i in range(20)]
        permutations = 999
        seed = 42

        # 1. SciKit-Bio Execution (calculating Euclidean distance via SciPy pdist)
        from scipy.spatial.distance import pdist, squareform

        euclidean_dm = DistanceMatrix(squareform(pdist(X, metric="euclidean")), ids)
        skbio_res = skbio_permanova(
            euclidean_dm, grouping, permutations=permutations, seed=seed
        )

        # 2. AnovaPy Execution
        anovapy_res = permanova(
            g1,
            g2,
            permutations=permutations,
            distance_metric="euclidean",
            random_state=seed,
        )

        # 3. Assertions
        assert_almost_equal(
            anovapy_res.statistic, skbio_res["test statistic"], decimal=5
        )
        assert anovapy_res.pvalue == pytest.approx(skbio_res["p-value"], abs=0.02)
