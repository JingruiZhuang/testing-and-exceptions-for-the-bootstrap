import numpy as np
import pytest

from studio04 import bootstrap_ci, r_squared


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
