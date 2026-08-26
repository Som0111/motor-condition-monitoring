# Bearing Fault Diagnosis via Envelope Spectrum Analysis

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

**Threshold calibration matters.** A fixed multiplier (3x noise floor) gives
75.2% with false alarms on healthy bearings — because a max-over-a-window
peak sits 2-3x above a median noise floor even on pure noise. Calibrating
the threshold from healthy-baseline data only (no fault data used) pushes it
to 83.2% with zero false alarms. This is basically how alarm limits get set
in the field.

**Ball faults don't show up in a fixed 2-5 kHz band.** Tried widening the
search — different bands, 2x the theoretical frequency — nothing found a
clean BSF line that a healthy bearing didn't also show. Also tried a
kurtogram (Antoni, 2006) to auto-select the best band per fault — it
correctly confirms inner/outer race resonances, but for the ball fault the
max spectral kurtosis is 0.33, an order of magnitude below the other two
faults, meaning there's genuinely no clean resonance to find here, not a
tuning problem. My guess is it comes down to the defect itself: a ball only
loads up when it rolls through the load zone, and each impact gets damped by
the cage and lubricant film before it can ring any single resonance sharply.
Inner and outer race defects sit in a fixed spot that gets struck the same
way every pass, so their impacts are more repeatable and excite a
resonance cleanly. A ball defect's contact geometry keeps changing as it
spins, so the energy ends up smeared across frequencies instead of
concentrated in one band. [TODO: replace with your own reasoning]

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
