# Pre-registration: confirmatory run

Written 18 November 2026, before the confirmatory run. Commit this file before
running, so its timestamp shows the rules came first.

## Why a second run

In the first run (1,000 questions), accuracy did not change for any published
edit. A positive control then showed that accuracy is a blunt measure here:
20% random noise on 30 matrices changed 36% of the model's answers and left
accuracy unchanged.

A sharper measure appeared in exploratory analysis: the probability the model
gives to the correct option, on questions the base model already answers
correctly with confidence. On that measure the broadest published edit
(`mit_p10c`) looked as costly as 20% noise, and the other edits looked cheap.
That measure was chosen after seeing the data, so it must be tested on new data.

## Data

2,000 MMLU-Pro test questions that were **not** among the 1,000 used in the
first run, drawn with a fixed seed (`run_retention.py --confirm 2000`). Model,
prompt, and scoring are unchanged.

## Variants

Base; eight distinct published edits (`ws_mmlupro` is left out because it is
identical to `mit_p25c`); the random-direction control; noise controls at 5%,
10%, 20%, 40% (seed 0) and at 10% and 20% (seed 1); layer-15 attention zeroed.

## Primary set and measure

- **Primary set:** questions where the base model gives the correct option a
  probability above 0.5. It is defined from the base model only.
- **Primary measure:** mean change in the probability of the correct option on
  that set, edited minus base, with a paired bootstrap 95% interval (10,000
  resamples).

Selecting questions by high base confidence means any perturbation tends to
lower the measure. The noise controls calibrate for this, so every claim is
made relative to noise, not relative to zero.

## Hypotheses

- **H1, calibration.** The drop grows with the noise level, and the 10% to 20%
  and 20% to 40% steps are each clear (interval of the difference excludes
  zero). If H1 fails, the measure is not usable and H2 and H3 are not interpreted.
- **H2, the broad edit.** `mit_p10c` lowers confidence more than 10% noise, for
  both noise seeds (interval of the difference excludes zero).
- **H3, the other edits.** Each of the other seven edits lowers confidence less
  than 20% noise, for both noise seeds.

## What each outcome means

- H1, H2, H3 pass: most published edits are close to free on firmly held
  knowledge, and the broadest one has a cost that accuracy hides.
- H2 fails: the first-run observation about `mit_p10c` was chance or specific to
  those 1,000 questions. It is reported as not replicated.
- H3 fails for some edit: that edit is named as more costly than expected.

## Secondary, descriptive only

Accuracy change with the first run's rules; share of the primary set kept
correct; the noise level that produces the same drop as each edit.

## Not covered

One model, one benchmark, one prompt format, thinking off. No plasticity test.
