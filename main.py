"""Vice Week One Forecaster: GTA VI trophy earn-rate forecasts for make.fun.

Method, in one paragraph: every trophy's earn rate is forecast on the logit
scale. Exophase refreshes its counts in steps, so consecutive reads are often
byte-identical; identical consecutive reads are dropped before fitting. Per
trophy, a least-squares line of logit(rate) against ln(1 + hours since the
2026-11-19T00:00:00Z release) is fitted over the distinct reads and read at
the target; each trophy's slope is shrunk toward its tier mean (platinum /
gold / silver / bronze) and the global mean, harder when there are few
distinct reads or the fit is poor, and a trophy with no usable trend falls
back to its last rate. The spread is deliberately honest and wide: a floor of about 1.15 logit
units, growing like a random walk with the square root of the
horizon in days, plus the fit's residual and slope-uncertainty terms, capped
so the bands stay sane. When the history is flat (fewer than 3 distinct
reads) the median is the last rate but the spread stays wide, because the
next Exophase refresh is likely to reprice every rate downward by dilution:
on launch night the first refresh took players tracked from 638 to 3,868 and
every rate but the 100% one fell.

Runs offline, Python 3.13 standard library only, deterministic: no
randomness, no wall-clock reads, files processed in name order.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Spread floor, in logit units: the honest uncertainty of a refresh-driven
# reprice, present even on a 90-minute horizon. The one refresh observed on
# launch night moved every rate down by roughly 1.8 logit units in a single
# step (players tracked 638 -> 3,868); a floor of 1.15 puts one such shock
# just inside the 5% tail (1.645 * 1.15 = 1.89), instead of just outside it.
SIGMA_FLOOR = 1.15
# Random-walk growth of the spread, logit units per sqrt(day).
SIGMA_RW = 0.5
# Hard cap on the total spread so the 5%-95% band never explodes.
SIGMA_CAP = 2.5
# How many distinct reads a slope needs before it is trusted on its own.
SLOPE_PRIOR_N = 5.0
# Cap on how far the fitted median may move from the last read, logit units.
MEDIAN_MOVE_CAP = 4.0


def clamp_rate(pct):
    """Clamp a percent into the range where logit is defined."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as the house baseline does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    delta = (moment - RELEASE).total_seconds() / 3600.0
    return max(0.0, delta)


def x_of(moment):
    """Log-time axis: ln(1 + hours since release)."""
    return math.log1p(hours_since_release(moment))


def distinct_points(history, index):
    """Distinct (x, logit rate) points for one trophy, deduped.

    Consecutive identical reads collapse to one point, and repeated
    (x, y) pairs are dropped, so a flat launch night contributes a single
    point instead of seven copies of the same number.
    """
    points = []
    seen = set()
    prev_rates = None
    for read in history:
        rates = read.get("rates") or []
        if rates == prev_rates:
            continue
        prev_rates = rates
        if index >= len(rates):
            continue
        rate = rates[index]
        if rate is None:
            continue
        try:
            moment = parse_time(read["at"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (round(x_of(moment), 9), round(logit(rate), 9))
        if key in seen:
            continue
        seen.add(key)
        points.append(key)
    points.sort()
    return points


def ols(points):
    """Least-squares line y = a + b*x. Returns (a, b, resid_var, x_mean)."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if sxx <= 1e-12:
        return my, 0.0, 0.0, mx
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    b = sxy / sxx
    a = my - b * mx
    resid = sum((p[1] - (a + b * p[0])) ** 2 for p in points)
    resid_var = resid / max(1, n - 2) if n > 2 else 0.0
    return a, b, resid_var, mx


def tier_slopes(fits):
    """Shrunk tier-mean slopes and the global mean slope.

    fits: list of (tier, slope, n_points, usable). A tier mean is pulled
    toward the global mean when the tier has few usable trophies.
    """
    usable = [f for f in fits if f[3]]
    if not usable:
        return {}, 0.0
    global_mean = sum(f[1] for f in usable) / len(usable)
    by_tier = {}
    for tier, slope, _n, _u in usable:
        by_tier.setdefault(tier, []).append(slope)
    means = {}
    for tier, slopes in by_tier.items():
        k = len(slopes)
        means[tier] = (sum(slopes) + 3.0 * global_mean) / (k + 3.0)
    return means, global_mean


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = x_of(target)

    last_rates = []
    if history:
        last = history[-1].get("rates") or []
        last_rates = list(last)
    while len(last_rates) < len(trophies):
        last_rates.append(None)

    # First pass: raw per-trophy fits, to learn the tier and global slopes.
    raw = []
    for i, trophy in enumerate(trophies):
        pts = distinct_points(history, i)
        usable = len(pts) >= 3 and (max(p[0] for p in pts) - min(p[0] for p in pts)) > 1e-9
        if usable:
            _a, b, rv, _mx = ols(pts)
        else:
            b, rv = 0.0, 0.0
        raw.append((trophy.get("tier", "bronze"), b, len(pts), usable, rv, pts))
    tier_mean, global_mean = tier_slopes(
        [(r[0], r[1], r[2], r[3]) for r in raw]
    )

    rows = []
    for i, trophy in enumerate(trophies):
        tier, slope, n_pts, usable, resid_var, pts = raw[i]
        last = last_rates[i]
        last_logit = logit(last) if last is not None else logit(1.0)

        if usable:
            # Shrink the slope toward its tier mean; a short or noisy fit
            # is trusted less. The tier mean itself is already shrunk
            # toward the global mean.
            w = n_pts / (n_pts + SLOPE_PRIOR_N)
            if resid_var > 0.0:
                w *= 1.0 / (1.0 + resid_var)
            slope_s = w * slope + (1.0 - w) * tier_mean.get(tier, global_mean)
            # Refit the intercept around the shrunk slope.
            intercept = sum(p[1] - slope_s * p[0] for p in pts) / len(pts)
            median_logit = intercept + slope_s * x_target
            # Slope uncertainty at the target, from the fit's scatter.
            x_mean = sum(p[0] for p in pts) / len(pts)
            sxx = sum((p[0] - x_mean) ** 2 for p in pts)
            slope_var = resid_var / sxx if sxx > 1e-12 else 0.0
            slope_term = slope_var * (x_target - x_mean) ** 2
            resid_term = resid_var
        else:
            # Flat or near-flat history: median is the last rate. The
            # spread below stays wide, because a refresh can reprice it.
            median_logit = last_logit
            slope_term = 0.0
            resid_term = 0.0

        # Never let the fitted median wander implausibly far from the
        # last read: a week of play moves rates, it does not teleport them.
        median_logit = min(last_logit + MEDIAN_MOVE_CAP,
                           max(last_logit - MEDIAN_MOVE_CAP, median_logit))

        sigma_sq = (SIGMA_FLOOR ** 2
                    + (SIGMA_RW * math.sqrt(horizon_days)) ** 2
                    + min(0.5, resid_term)
                    + min(0.5, slope_term))
        sigma = min(SIGMA_CAP, math.sqrt(sigma_sq))

        qs = []
        for z in Z:
            q = round4(clamp_rate(inv_logit(median_logit + z * sigma)))
            qs.append(q)
        # Rounding can only move a value by 0.00005, but enforce the
        # non-decreasing order the format requires regardless.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
