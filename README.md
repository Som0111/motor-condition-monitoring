# Bearing Fault Diagnosis via Envelope Spectrum Analysis

Part of a broader motor condition monitoring toolkit — this piece covers
vibration-based bearing fault detection.

Diagnosing rolling-element bearing faults from raw vibration signals — no
training, no deep learning. Uses envelope (Hilbert) spectrum analysis to pull
out fault frequencies that don't show up in a plain FFT. A RandomForest is
included only as a cross-check on the extracted features.

Data: [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file),
drive-end accelerometer, 12 kHz, 0.007" defects, ~1797 rpm.

```
python bearing_diagnosis.py
```

## Why

Bearing failures are one of the most common ways induction motors actually
die in the field, and they don't announce themselves until it's often too
late. Vibration monitoring is one of the few ways to catch it early — while
it's still a scheduled bearing swap, not a breakdown.

## Method

Bearing: SKF 6205-2RS (n=9 balls, d=0.3126", D=1.537", contact angle 0°).
Shaft speed is read from each file's own RPM field, not assumed — theoretical
defect frequencies (BPFO, BPFI, BSF, FTF) are computed from that, not
fitted to the data.

A plain FFT of the raw signal doesn't show these — the defect energy shows
up as amplitude modulation on a high-frequency structural resonance, not as
a clean tone. So the pipeline is: bandpass 2–5 kHz → Hilbert transform →
envelope → FFT of the envelope. That recovers the actual fault repetition
rate.

![envelope spectrum](plots/envelope_spectrum_diagnosis.png)

Six features per 1-second window: RMS, kurtosis, crest factor, and envelope
amplitude at BPFO/BPFI/BSF. Each fault lights up its own frequency by 25–2700x
over the healthy baseline — see `results/feature_dataset.csv`.

## Results

**Rule-based diagnosis (primary method):** compare each defect line's
amplitude to the local noise floor, flag whichever clears it by the largest
margin. Zero training required.

**83.2% accuracy, zero false alarms on healthy bearings.** Inner race and
outer race are both 100%. All errors are the ball fault being missed.

![confusion matrix](plots/confusion_matrix_rulebased.png)

**RandomForest cross-check:** 100% accuracy, and the top 3 features by
importance are the three envelope-amplitude-at-defect-frequency features —
confirms the physics-based features actually carry the signal, this isn't
just noise the model happened to fit.

## Two things worth flagging

First one: I initially just used a flat 3x-the-noise-floor cutoff for the
rule-based check, and it only got 75.2%, with a bunch of healthy bearings
flagged as faulty. Turns out that's just how the math works out — a peak
picked as the max over a window naturally sits 2-3x above a median noise
floor even when there's no fault at all, so 3x was never a fair cutoff to
begin with. Once I calibrated the threshold off healthy data only (never
touching the fault windows), accuracy went to 83.2% and the false alarms
disappeared. Which, now that I think about it, is basically how you'd set
an alarm limit on a real machine anyway — you don't guess a number, you
look at what "normal" actually looks like first.

Second, and this one took longer to accept: the ball fault just doesn't
show up cleanly in a fixed 2-5 kHz band, no matter what I did. I tried
other bands, I tried looking at 2x the expected frequency in case I had the
harmonic wrong, nothing gave me a BSF peak that a healthy bearing wasn't
also showing. So I brought in a kurtogram (Antoni, 2006) to let the
algorithm pick the best band per fault instead of me guessing one band for
everything. It nailed the inner and outer race cases, but for the ball
fault the best band it could find had a spectral kurtosis of 0.33 — an
order of magnitude worse than the other two. That's not a tuning issue,
that's the algorithm telling me there's no clean resonance to find. My
best guess for why: a ball defect only makes contact when it rolls through
the load zone, and the cage and the lubricant film soak up a lot of that
impact before it can ring anything sharply. Inner and outer race defects
get struck at the same spot every single pass, so they're more repeatable
and excite a resonance cleanly, whereas a ball's contact point keeps
shifting as it spins, so the energy just smears out instead of piling up
in one place.

## What I'd add with more time

The ball fault needs a different feature entirely — RMS/kurtosis trend
alarms (which is what the RandomForest is actually using to catch it) or a
different signal processing approach, not a better frequency band.

## Files

```
bearing_diagnosis.py       everything - loading, features, both methods
plots/                     all figures referenced above
results/                   defect frequency table, extracted features (csv)
```
