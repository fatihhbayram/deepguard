"""The confidence bounds: the closed forms they must reproduce, and the zero they must not print."""

import math

import pytest

from eval import stats


def test_zero_observed_is_never_reported_as_a_zero_rate():
    """The whole reason this module exists.

    Zero events in three hundred draws is not a zero rate, and the bound is what says so.
    """
    rate = stats.rate(0, 300)
    assert rate.observed == 0.0
    assert rate.upper_95 > 0.0


def test_the_zero_count_bound_matches_its_closed_form():
    """`k = 0` has an exact solution, and the general path must agree with it."""
    for n in (1, 20, 54, 299, 1000):
        assert stats.clopper_pearson_upper(0, n) == pytest.approx(
            1 - 0.05 ** (1 / n), rel=1e-12
        )


def test_the_acceptance_target_sample_size_is_299():
    """Where the corpus's genuine evaluation size comes from, checked rather than asserted."""
    assert stats.required_n_for_zero_observed(0.01) == 299
    assert stats.clopper_pearson_upper(0, 299) <= 0.01
    assert stats.clopper_pearson_upper(0, 298) > 0.01


def test_bounds_against_known_beta_quantiles():
    """Clopper-Pearson is a Beta quantile; these are the values a statistics table gives.

    `k = 3, n = 100` upper is the 0.95 quantile of Beta(4, 97); `k = 20, n = 20` lower is
    `0.05 ** (1 / 20)`.
    """
    assert stats.clopper_pearson_upper(3, 100) == pytest.approx(0.0757107937, rel=1e-8)
    assert stats.clopper_pearson_lower(20, 20) == pytest.approx(0.05 ** (1 / 20), rel=1e-12)
    assert stats.clopper_pearson_upper(1, 302) == pytest.approx(0.0156110, rel=1e-5)


def test_the_incomplete_beta_is_a_cdf():
    """Monotone, bounded, and symmetric in the way the identity `I_x(a,b) = 1 - I_1-x(b,a)` says."""
    for x in (0.01, 0.2, 0.5, 0.8, 0.99):
        left = stats.regularised_incomplete_beta(3, 7, x)
        right = stats.regularised_incomplete_beta(7, 3, 1 - x)
        assert left == pytest.approx(1 - right, abs=1e-12)
        assert 0.0 <= left <= 1.0
    values = [stats.regularised_incomplete_beta(4, 9, x / 20) for x in range(21)]
    assert values == sorted(values)


def test_an_empty_denominator_is_undefined_and_not_zero():
    rate = stats.rate(0, 0)
    assert rate.observed is None
    assert rate.upper_95 is None
    assert "undefined" in rate.describe()


def test_every_flagged_gives_a_bound_of_one():
    assert stats.clopper_pearson_upper(50, 50) == 1.0


def test_describe_states_the_count_the_rate_and_the_bound():
    """A reported rate that does not carry its `n` is the failure mode being designed out."""
    text = stats.rate(0, 302).describe()
    assert "0/302" in text
    assert "upper bound" in text


def test_the_bound_tightens_as_the_sample_grows():
    bounds = [stats.clopper_pearson_upper(0, n) for n in (20, 54, 150, 302)]
    assert bounds == sorted(bounds, reverse=True)
    assert bounds[0] > 0.13  # R5-T3's twenty genuine clips
    assert bounds[-1] < 0.01
