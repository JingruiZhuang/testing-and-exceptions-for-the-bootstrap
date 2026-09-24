import numpy as np
import pytest

from studio04 import bootstrap_sample, bootstrap_ci, r_squared


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------

class TestBootstrapCI:

    # --- happy path ---

    def test_returns_tuple_of_two(self):
        stats = np.random.default_rng(0).normal(size=1000)
        ci = bootstrap_ci(stats)
        assert isinstance(ci, tuple)
        assert len(ci) == 2

    def test_lower_not_above_upper(self):
        stats = np.random.default_rng(1).normal(size=1000)
        lower, upper = bootstrap_ci(stats)
        assert lower <= upper

    def test_percentiles_of_uniform_grid(self):
        # For evenly spaced values on [0, 1], the 95% CI is about (0.025, 0.975)
        stats = np.linspace(0, 1, 10001)
        lower, upper = bootstrap_ci(stats, alpha=0.05)
        assert lower == pytest.approx(0.025, abs=1e-3)
        assert upper == pytest.approx(0.975, abs=1e-3)

    def test_matches_normal_quantiles(self):
        stats = np.random.default_rng(2).normal(loc=3, scale=2, size=200_000)
        lower, upper = bootstrap_ci(stats, alpha=0.05)
        assert lower == pytest.approx(3 - 1.96 * 2, abs=0.05)
        assert upper == pytest.approx(3 + 1.96 * 2, abs=0.05)

    def test_bounds_within_data_range(self):
        stats = np.random.default_rng(3).exponential(size=500)
        lower, upper = bootstrap_ci(stats)
        assert stats.min() <= lower <= upper <= stats.max()

    def test_smaller_alpha_gives_wider_interval(self):
        stats = np.random.default_rng(4).normal(size=5000)
        lo90, hi90 = bootstrap_ci(stats, alpha=0.10)
        lo99, hi99 = bootstrap_ci(stats, alpha=0.01)
        assert lo99 <= lo90
        assert hi99 >= hi90

    def test_order_of_stats_does_not_matter(self):
        stats = np.random.default_rng(5).normal(size=1000)
        shuffled = np.random.default_rng(6).permutation(stats)
        lower1, upper1 = bootstrap_ci(stats)
        lower2, upper2 = bootstrap_ci(shuffled)
        assert lower1 == pytest.approx(lower2)
        assert upper1 == pytest.approx(upper2)

    def test_does_not_modify_input(self):
        stats = np.random.default_rng(7).normal(size=100)
        original = stats.copy()
        lower, upper = bootstrap_ci(stats)
        np.testing.assert_array_equal(stats, original)

    # --- edge cases ---

    def test_constant_stats(self):
        stats = np.full(100, 4.2)
        lower, upper = bootstrap_ci(stats)
        assert lower == pytest.approx(4.2)
        assert upper == pytest.approx(4.2)

    def test_single_stat(self):
        lower, upper = bootstrap_ci(np.array([1.5]))
        assert lower == pytest.approx(1.5)
        assert upper == pytest.approx(1.5)

    def test_alpha_close_to_boundaries(self):
        stats = np.random.default_rng(8).normal(size=1000)
        lo_narrow, hi_narrow = bootstrap_ci(stats, alpha=0.999)
        lo_wide, hi_wide = bootstrap_ci(stats, alpha=0.001)
        assert lo_narrow <= hi_narrow
        assert lo_wide <= hi_wide

    # --- invalid inputs ---

    @pytest.mark.parametrize("alpha", [0, 1, -0.05, 1.5])
    def test_alpha_out_of_range_raises(self, alpha):
        stats = np.random.default_rng(9).normal(size=100)
        with pytest.raises(ValueError):
            bootstrap_ci(stats, alpha=alpha)

    def test_empty_stats_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([]))


# ---------------------------------------------------------------------------
# r_squared
# ---------------------------------------------------------------------------

class TestRSquared:

    # --- happy path ---

    def test_returns_float(self):
        data = np.random.default_rng(10).normal(size=(50, 2))
        assert isinstance(r_squared(data), float)

    def test_perfect_positive_line(self):
        x = np.arange(10.0)
        data = np.column_stack([x, 3 * x + 2])
        assert r_squared(data) == pytest.approx(1.0)

    def test_perfect_negative_line(self):
        x = np.arange(10.0)
        data = np.column_stack([x, -0.5 * x + 7])
        assert r_squared(data) == pytest.approx(1.0)

    def test_hand_computed_value(self):
        # x = [1, 2, 3], y = [1, 3, 2]: fitted line y = 1.5 + 0.5x
        # SSE = 1.5, SST = 2, so R^2 = 1 - 1.5 / 2 = 0.25
        data = np.array([[1, 1], [2, 3], [3, 2]])
        assert r_squared(data) == pytest.approx(0.25)

    def test_matches_squared_correlation(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=200)
        y = 0.7 * x + rng.normal(size=200)
        expected = np.corrcoef(x, y)[0, 1] ** 2
        assert r_squared(np.column_stack([x, y])) == pytest.approx(expected)

    def test_between_zero_and_one(self):
        rng = np.random.default_rng(12)
        for _ in range(20):
            data = rng.normal(size=(30, 2))
            assert 0 <= r_squared(data) <= 1

    def test_symmetric_in_x_and_y(self):
        # In simple linear regression, R^2 is the same if x and y are swapped
        rng = np.random.default_rng(13)
        data = rng.normal(size=(100, 2))
        r2 = r_squared(data)
        assert isinstance(r2, float)
        assert r2 == pytest.approx(r_squared(data[:, ::-1]))

    def test_invariant_to_scale_and_shift(self):
        rng = np.random.default_rng(14)
        data = rng.normal(size=(100, 2))
        transformed = data * np.array([10.0, -3.0]) + np.array([5.0, 100.0])
        r2 = r_squared(data)
        assert isinstance(r2, float)
        assert r2 == pytest.approx(r_squared(transformed))

    def test_accepts_list_of_lists(self):
        data = [[1, 1], [2, 3], [3, 2]]
        assert r_squared(data) == pytest.approx(0.25)

    # --- edge cases ---

    def test_two_points_fit_exactly(self):
        data = np.array([[0.0, 1.0], [1.0, 5.0]])
        assert r_squared(data) == pytest.approx(1.0)

    def test_unrelated_data_gives_small_r_squared(self):
        rng = np.random.default_rng(15)
        data = rng.normal(size=(10_000, 2))
        assert r_squared(data) < 0.01

    # --- invalid inputs ---

    @pytest.mark.parametrize("shape", [(10, 1), (10, 3)])
    def test_wrong_number_of_columns_raises(self, shape):
        with pytest.raises(ValueError):
            r_squared(np.ones(shape))

    def test_one_dimensional_input_raises(self):
        with pytest.raises(ValueError):
            r_squared(np.arange(10.0))

    def test_single_row_raises(self):
        with pytest.raises(ValueError):
            r_squared(np.array([[1.0, 2.0]]))

    def test_empty_data_raises(self):
        with pytest.raises(ValueError):
            r_squared(np.empty((0, 2)))


# ---------------------------------------------------------------------------
# bootstrap_sample (Student B)
# ---------------------------------------------------------------------------

class TestBootstrapSample:

    def test_output_length_and_type(self):
        np.random.seed(20)
        stats = bootstrap_sample([1, 2, 3, 4], np.mean, n_bootstrap=25)
        assert isinstance(stats, np.ndarray)
        assert stats.shape == (25,)
        assert np.all((stats >= 1) & (stats <= 4))

    def test_constant_data_stays_constant(self):
        np.random.seed(21)
        stats = bootstrap_sample([7, 7, 7], np.mean, n_bootstrap=30)
        np.testing.assert_array_equal(stats, np.full(30, 7.0))

    def test_samples_rows_with_replacement_and_keeps_pairs_together(self):
        data = np.column_stack((np.arange(10), 3 * np.arange(10) + 1))
        seen_duplicate = False

        def check_sample(sample):
            nonlocal seen_duplicate
            assert sample.shape == data.shape
            np.testing.assert_array_equal(sample[:, 1], 3 * sample[:, 0] + 1)
            seen_duplicate |= len(np.unique(sample[:, 0])) < len(sample)
            return np.mean(sample[:, 0])

        np.random.seed(22)
        bootstrap_sample(data, check_sample, n_bootstrap=15)
        assert seen_duplicate

    def test_reproducible_with_same_seed(self):
        np.random.seed(23)
        first = bootstrap_sample([1, 2, 3, 4], np.mean, n_bootstrap=10)
        np.random.seed(23)
        second = bootstrap_sample([1, 2, 3, 4], np.mean, n_bootstrap=10)
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("data", [[], np.empty((0, 2)), np.zeros((2, 2, 2))])
    def test_rejects_empty_or_invalid_dimensions(self, data):
        with pytest.raises(ValueError):
            bootstrap_sample(data, np.mean, n_bootstrap=5)

    @pytest.mark.parametrize("n_bootstrap", [0, -1])
    def test_rejects_nonpositive_bootstrap_count(self, n_bootstrap):
        with pytest.raises(ValueError):
            bootstrap_sample([1, 2], np.mean, n_bootstrap=n_bootstrap)

    def test_rejects_noncallable_statistic(self):
        with pytest.raises(TypeError):
            bootstrap_sample([1, 2], None, n_bootstrap=5)


def test_bootstrap_r_squared_confidence_interval_integration():
    x = np.arange(40, dtype=float)
    y = 2 * x + np.sin(x) * 3
    data = np.column_stack((x, y))

    np.random.seed(24)
    bootstrap_stats = bootstrap_sample(data, r_squared, n_bootstrap=300)
    lower, upper = bootstrap_ci(bootstrap_stats, alpha=0.05)

    assert bootstrap_stats.shape == (300,)
    assert 0 <= lower <= upper <= 1
    assert lower <= r_squared(data) <= upper


def test_null_distribution_of_r_squared_has_expected_mean():
    # Bonus: for independent normal x and y with n observations,
    # R^2 follows Beta(1/2, (n-2)/2), whose mean is 1/(n-1).
    rng = np.random.default_rng(25)
    n = 30
    observed = np.array([
        r_squared(rng.normal(size=(n, 2))) for _ in range(2000)
    ])
    assert np.mean(observed) == pytest.approx(1 / (n - 1), abs=0.005)
