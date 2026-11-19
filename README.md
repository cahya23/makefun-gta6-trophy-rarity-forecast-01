# Vice Week One Forecaster

[![make.fun badge](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun rank](https://make.fun/rank/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![make.fun reward](https://make.fun/reward/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for every Grand Theft Auto VI trophy on PlayStation 5, its earn rate
on Exophase at 2026-11-26T00:00:00Z, a week after launch, as five quantiles
(5%, 25%, 50%, 75%, 95%) per trophy.

## Method

The forecast is made on the logit scale. Exophase refreshes its counts in
steps, so consecutive reads of the trophy page are often identical; identical
consecutive reads are dropped before fitting, so a flat launch night counts
as one observation instead of many copies of the same number. For each
trophy, a least-squares line of logit(rate) against ln(1 + hours since the
2026-11-19T00:00:00Z release) is fitted over the distinct reads and read at
the target. Each trophy's slope is shrunk toward its tier mean (platinum,
gold, silver, bronze) and the global mean, harder when there are few distinct
reads or the fit is poor; a trophy with no usable trend falls back to its
last rate. The spread is deliberately honest and wide: a floor of about 1.15
logit units — sized so one launch-night-style refresh shock (players tracked
638 to 3,868, every rate down roughly 1.8 logit units in a single step) sits
just inside the 5% tail — growing like a random walk with the square root of
the horizon in days, plus the fit's residual and slope-uncertainty terms,
capped so the bands stay sane. When the history is flat the median is the
last rate but the spread stays wide, because the next refresh is likely to
reprice every rate downward by dilution.

## How to run

Python 3.13, standard library only, no network. Reads every `*.in` case in
the input directory and writes one `<id>.out` per case to the output
directory:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Without those variables it reads `/input` and writes `/output`, as the
grader runs it. The program is deterministic: two runs on the same case
write byte-identical output, and a run takes well under a second on the
practice cases (limit: 8 seconds per case).

## Practice check

On the kit's practice cases (`p00`, `p01`) the output validates cleanly:
55 rows per case, one per trophy, five non-decreasing quantiles each in
0–100, deterministic across runs. Under a holders-constant dilution guess
for the practice answers (last rate × 638/3868, the 100% trophy unchanged)
it scores a mean pinball loss of about 0.99, against about 1.69 for the
house persistence baseline under the same guess — but that guess is only a
sanity check, not a tuning target: the final case carries two days of real
reads, and the trend fit is what matters there.
