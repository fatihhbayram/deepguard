"""Confidence bounds for observed rates, with no dependency and no fabricated precision.

The one number this task must never print is `0%`. A detector that flagged none of three hundred
genuine lineages has not been shown to have a zero false-positive rate; it has been shown to have
one small enough that three hundred draws could plausibly miss it, and the size of "small enough"
is what an upper bound states and a point estimate hides.

**Clopper-Pearson, exact, one-sided.** The interval is inverted from the binomial distribution
itself rather than from a normal approximation, because the approximation is worst exactly where
this task lives — small counts near zero, where Wald's interval is famously degenerate and
returns `[0, 0]` for `k = 0`. Exact means conservative: the true coverage is at least 95%, never
less.

The bound is computed from the Beta quantile, using the standard identity that the Clopper-Pearson
upper limit for `k` successes in `n` trials is the `1 - alpha` quantile of `Beta(k + 1, n - k)`.
`k = 0` collapses to the closed form `1 - alpha^(1/n)`, which is where the acceptance target's
`n = 299` comes from.

Standard library only: `math.lgamma` supplies the log-Beta, the regularised incomplete Beta is a
continued fraction (Lentz's method, as in Numerical Recipes §6.4), and the quantile is a bisection
over it. Bisection rather than Newton because the function is monotone on `[0, 1]` and 200
bisections give ~60 bits of precision — far more than a rate reported to four decimals needs, at
a cost of microseconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Iterations for the continued fraction, and the relative change below which it has converged.
# Both are the conventional values; the fraction converges in tens of iterations for the
# arguments this module sees.
_MAX_ITERATIONS = 300
_EPSILON = 1e-14

# Bisection steps for the quantile. 200 halvings of [0, 1] is well past double precision.
_BISECTIONS = 200


@dataclass(frozen=True)
class Rate:
    """An observed rate and what can honestly be said about the rate behind it.

    `observed` is `numerator / denominator` and is a description of the sample. `upper_95` is a
    statement about the population, and it is the one a decision should be made on. `None` for
    both when the denominator is zero: a rate over no observations is undefined, not zero.
    """

    numerator: int
    denominator: int
    observed: float | None
    upper_95: float | None
    lower_95: float | None

    def as_dict(self) -> dict:
        return {
            "k": self.numerator,
            "n": self.denominator,
            "observed": self.observed,
            "upper_95_one_sided": self.upper_95,
            "lower_95_one_sided": self.lower_95,
        }

    def describe(self) -> str:
        """One line for a report table: the count, the rate, and the bound that qualifies it."""
        if self.denominator == 0:
            return "n = 0 (undefined)"
        return (
            f"{self.numerator}/{self.denominator} = {self.observed:.4%} "
            f"(one-sided 95% upper bound {self.upper_95:.4%})"
        )


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction for the incomplete Beta, by the modified Lentz method."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITERATIONS + 1):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPSILON:
            break
    return h


def regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """`I_x(a, b)`, the CDF of `Beta(a, b)` at `x`."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        a * math.log(x) + b * math.log1p(-x) - _log_beta(a, b)
    ) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    mirror = math.exp(
        b * math.log1p(-x) + a * math.log(x) - _log_beta(b, a)
    ) / b
    return 1.0 - mirror * _betacf(b, a, 1.0 - x)


def beta_quantile(a: float, b: float, probability: float) -> float:
    """The `probability` quantile of `Beta(a, b)`, by bisection over its monotone CDF."""
    low, high = 0.0, 1.0
    for _ in range(_BISECTIONS):
        middle = (low + high) / 2.0
        if regularised_incomplete_beta(a, b, middle) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float | None:
    """The exact one-sided upper confidence bound on a binomial rate.

    `k = n` returns 1.0 and `k = 0` returns the closed form `1 - (1 - confidence)^(1/n)`, which
    is the case the acceptance target is written against.
    """
    if n <= 0:
        return None
    if k >= n:
        return 1.0
    if k == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / n)
    return beta_quantile(k + 1, n - k, confidence)


def clopper_pearson_lower(k: int, n: int, confidence: float = 0.95) -> float | None:
    """The exact one-sided lower confidence bound, for the rates where the floor is what matters.

    Reported beside the upper bound for detection rates, where a reader needs to know how low
    the true rate could be, not how high.
    """
    if n <= 0:
        return None
    if k <= 0:
        return 0.0
    if k >= n:
        return (1.0 - confidence) ** (1.0 / n)
    return beta_quantile(k, n - k + 1, 1.0 - confidence)


def rate(k: int, n: int) -> Rate:
    """One observed rate with both exact one-sided 95% bounds, or an undefined rate at `n = 0`."""
    if n <= 0:
        return Rate(k, n, None, None, None)
    return Rate(
        k,
        n,
        k / n,
        clopper_pearson_upper(k, n),
        clopper_pearson_lower(k, n),
    )


def required_n_for_zero_observed(target_upper: float, confidence: float = 0.95) -> int:
    """The smallest `n` at which zero observed events supports an upper bound at or below target.

    The inverse of the `k = 0` closed form, and the arithmetic behind the acceptance target's
    "approximately 300": at `target_upper = 0.01` it returns 299.
    """
    if not 0.0 < target_upper < 1.0:
        raise ValueError("target_upper must lie strictly between 0 and 1")
    exact = math.log(1.0 - confidence) / math.log(1.0 - target_upper)
    n = max(1, math.floor(exact))
    while clopper_pearson_upper(0, n, confidence) > target_upper:
        n += 1
    return n
