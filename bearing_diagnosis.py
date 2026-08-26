"""
Bearing fault diagnosis on CWRU drive-end accelerometer data via envelope
(Hilbert) spectrum analysis.

Pipeline: load -> theoretical defect frequencies -> raw FFT (shows failure)
-> envelope spectrum -> physics features -> rule-based diagnosis -> RF cross-check.

Run: python bearing_diagnosis.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

FS = 12000                 # sampling rate, Hz
WINDOW = 12000             # 1-second windows -> 1 Hz spectral resolution
# ponytail: 90% overlap. Non-overlapping windows yield only 10-20 per class from
# these files, far below the 150-200 target. Set HOP = WINDOW for strictly
# independent windows (and an honest ML split) at the cost of sample count.
HOP = 1200
BAND = (2000, 5000)        # fixed resonance band for envelope demodulation
# Kurtogram search limits. fmin skips the shaft-order content that wins on
# kurtosis without being a resonance; fmax keeps the order-4 bandpass off
# Nyquist, where the filter design degenerates and fakes kurtosis in the 100s.
# min_bw = 2 x the 500 Hz envelope range: a W-wide band only carries envelope
# content up to ~W/2, so anything narrower leaves the noise floor empty.
KURT_LEVELS, KURT_FMIN, KURT_FMAX, KURT_MIN_BW = 6, 1000.0, 5400.0, 1000.0
TOL = 2.0                  # +-Hz tolerance when reading a defect-frequency peak
DATA_DIR = "cwru_data"
PLOTS, RESULTS = "plots", "results"

# SKF 6205-2RS drive-end bearing geometry (inches)
N_BALLS, D_BALL, D_PITCH, CONTACT_ANGLE = 9, 0.3126, 1.537, 0.0

FILES = {
    "Normal":     ("normal.mat",     "X097_DE_time", "X097RPM"),
    "Inner Race": ("inner_race.mat", "X105_DE_time", "X105RPM"),
    "Ball":       ("ball.mat",       "X118_DE_time", "X118RPM"),
    "Outer Race": ("outer_race.mat", "X130_DE_time", "X130RPM"),
}
CLASSES = list(FILES)


# ---------------------------------------------------------------- step 1
def load_class(fname, sig_key, rpm_key):
    """Return (drive-end signal 1-D, shaft frequency Hz) for one .mat file."""
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing - download from CWRU, do not synthesise.")
    mat = loadmat(path)
    signal = np.asarray(mat[sig_key], dtype=float).ravel()
    rpm = float(np.asarray(mat[rpm_key]).ravel()[0])   # measured, not assumed 1797
    return signal, rpm / 60.0


def segment(signal, window=WINDOW, hop=HOP):
    """Slice a signal into fixed-length windows -> array (n_windows, window)."""
    starts = range(0, len(signal) - window + 1, hop)
    return np.array([signal[s:s + window] for s in starts])


# ---------------------------------------------------------------- step 2
def bearing_defect_frequencies(fr, n, d, D, contact_angle_deg=0.0):
    """Standard rolling-element bearing kinematics (ISO 15243). All values in Hz.

    fr: shaft rotation frequency, n: ball count, d: ball diameter,
    D: pitch diameter, contact_angle_deg: contact angle in degrees.
    """
    c = np.cos(np.radians(contact_angle_deg))
    return {
        "BPFO": (n / 2) * fr * (1 - (d / D) * c),
        "BPFI": (n / 2) * fr * (1 + (d / D) * c),
        "BSF":  (D / (2 * d)) * fr * (1 - (d / D) ** 2 * c ** 2),
        "FTF":  (fr / 2) * (1 - (d / D) * c),
    }


# ---------------------------------------------------------------- step 3/4
def raw_spectrum(window, fs=FS):
    """Single-sided amplitude spectrum of a raw window."""
    n = len(window)
    mag = 2 * np.abs(np.fft.rfft(window - window.mean())) / n
    return np.fft.rfftfreq(n, 1 / fs), mag


def envelope_spectrum(signal, fs=FS, band=BAND, fmax=500.0):
    """Bandpass -> Hilbert envelope -> DC-removed FFT. Returns (freqs, magnitude).

    Defect impacts ring the structural resonance inside `band`; the impact rate
    lives in the amplitude modulation, not in the raw spectrum.
    """
    b, a = butter(4, [band[0] / (fs / 2), band[1] / (fs / 2)], btype="bandpass")
    env = np.abs(hilbert(filtfilt(b, a, signal)))
    env -= env.mean()
    n = len(env)
    mag = 2 * np.abs(np.fft.rfft(env)) / n
    freqs = np.fft.rfftfreq(n, 1 / fs)
    keep = freqs <= fmax
    return freqs[keep], mag[keep]


def peak_at(freqs, mag, target, tol=TOL):
    """Largest envelope magnitude within +-tol Hz of a theoretical frequency."""
    band = np.abs(freqs - target) <= tol
    return float(mag[band].max()) if band.any() else 0.0


def baseline_noise(freqs, mag, defect_freqs, guard=5.0):
    """Median envelope magnitude excluding +-guard Hz around each defect line."""
    mask = np.ones_like(freqs, dtype=bool)
    for f in defect_freqs:
        mask &= np.abs(freqs - f) > guard
    return float(np.median(mag[mask]))


# ------------------------------------------------- kurtogram band selection
# Antoni, J. (2006). "The spectral kurtosis: a useful tool for characterising
# non-stationary signals." Mechanical Systems and Signal Processing, 20(2), 282-307.
def kurtogram(signal, fs=FS, levels=KURT_LEVELS, fmin=KURT_FMIN,
              fmax=KURT_FMAX, min_bw=KURT_MIN_BW):
    """Binary-tree filter bank -> spectral kurtosis per band.

    Returns [(level, f_lo, f_hi, SK), ...]. The band with maximum SK carries the
    most impulsive content, i.e. the resonance a defect is ringing. Implemented
    as an explicit scipy filter bank rather than a wavelet packet tree: it reuses
    the exact butter+hilbert path the rest of the pipeline uses, so the selected
    band is directly comparable to the fixed 2-5 kHz one, and it is ~40 filter
    passes on 12k samples - fast enough.

    SK is Antoni's definition on the complex envelope c, <|c|^4>/<|c|^2>^2 - 2
    (0 for a Gaussian band). NOT kurtosis(|c|), which looks equivalent and is
    not: it systematically selects the band NEXT to the resonance, because a
    transient leaking through a steep filter skirt is sparser - and so scores
    higher on plain kurtosis - than the sustained ring inside the true band.

    Band structure is Antoni's 1/3-binary tree: the spectrum is split into
    n = 2, 3, 4, 6, 8, 12, ... equal bands, which gives finer resolution than
    powers of two alone.

    fmin excludes the deterministic shaft-order content below 1 kHz, which
    otherwise wins on kurtosis without being a resonance; fmax keeps the order-4
    bandpass away from Nyquist, where its design goes degenerate and reports
    kurtosis in the hundreds from pure numerical artefact.

    min_bw is a hard requirement, not a tuning knob: demodulating a band of
    width W gives an envelope with content only up to ~W/2, so reading defect
    lines out to `fmax` of envelope_spectrum needs W >= 2x that. Narrower bands
    leave the upper envelope spectrum empty, which collapses the median noise
    floor and makes every peak look enormous.
    """
    out = []
    for n_bands in [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64][:2 * levels]:
        bw = (fs / 2) / n_bands
        if bw < min_bw:
            break
        for i in range(n_bands):
            lo, hi = i * bw, (i + 1) * bw
            if lo < fmin or hi > fmax:
                continue
            b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="bandpass")
            env = np.abs(hilbert(filtfilt(b, a, signal)))
            sk = np.mean(env ** 4) / np.mean(env ** 2) ** 2 - 2
            out.append((bw, lo, hi, float(sk)))
    if not out:
        raise ValueError(f"no candidate band satisfies fmin={fmin}, fmax={fmax}, "
                         f"min_bw={min_bw}")
    return out


def select_band(signal, **kw):
    """Auto-select the demodulation band: argmax spectral kurtosis. -> (lo, hi)."""
    bw, lo, hi, k = max(kurtogram(signal, **kw), key=lambda r: r[3])
    return (lo, hi), k


def plot_kurtogram(kmap, selected, title, path):
    """Heatmap of spectral kurtosis over (frequency, demodulation bandwidth)."""
    bws = sorted({r[0] for r in kmap})
    grid = np.full((len(bws), 600), np.nan)
    edges = np.linspace(0, FS / 2, 601)
    for bw, lo, hi, k in kmap:
        cols = (edges[:-1] >= lo) & (edges[:-1] < hi)
        grid[bws.index(bw), cols] = k
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                   interpolation="nearest",
                   extent=[0, FS / 2, -0.5, len(bws) - 0.5])
    (lo, hi), k = selected
    ax.add_patch(plt.Rectangle((lo, -0.5), hi - lo, len(bws),
                               fill=False, edgecolor="red", lw=2))
    ax.set_yticks(range(len(bws)))
    ax.set_yticklabels([f"{b:.0f}" for b in bws])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("demodulation bandwidth (Hz)")
    ax.set_title(f"{title}\nselected {lo:.0f}-{hi:.0f} Hz, max spectral kurtosis = {k:.2f}")
    fig.colorbar(im, ax=ax, label="spectral kurtosis")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- step 5
def extract_features(windows, defect, label, band=BAND):
    """One row per window: 3 time-domain + 3 envelope-at-defect-frequency features."""
    rows = []
    for w in windows:
        rms = float(np.sqrt(np.mean(w ** 2)))
        freqs, mag = envelope_spectrum(w, band=band)
        targets = (defect["BPFO"], defect["BPFI"], defect["BSF"])
        rows.append({
            "rms": rms,
            "kurtosis": float(kurtosis(w)),
            "crest_factor": float(np.max(np.abs(w)) / rms),
            "env_amp_bpfo": peak_at(freqs, mag, defect["BPFO"]),
            "env_amp_bpfi": peak_at(freqs, mag, defect["BPFI"]),
            "env_amp_bsf":  peak_at(freqs, mag, defect["BSF"]),
            "baseline_noise": baseline_noise(freqs, mag, targets),
            "label": label,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- step 6
def diagnose_fault(features_row, noise_floor_threshold=3.0):
    """Rule-based, zero-training diagnosis.

    Whichever defect line clears the local envelope noise floor by the largest
    margin wins; if none clears it, the bearing is healthy.
    """
    ratios = {
        "Outer Race": features_row["env_amp_bpfo"] / features_row["baseline_noise"],
        "Inner Race": features_row["env_amp_bpfi"] / features_row["baseline_noise"],
        "Ball":       features_row["env_amp_bsf"] / features_row["baseline_noise"],
    }
    best = max(ratios, key=ratios.get)
    return best if ratios[best] > noise_floor_threshold else "Normal"


def calibrate_threshold(df, healthy="Normal", margin=1.2):
    """Alarm threshold set from the healthy baseline only - no fault data used.

    A peak read as max-over-+-2Hz sits ~2-3x above a median noise floor even on
    pure noise, so the nominal 3.0 alarms on healthy bearings. Field practice is
    to take the worst ratio ever seen on a known-good machine and add margin.
    """
    h = df[df["label"] == healthy]
    worst = max((h[c] / h["baseline_noise"]).max()
                for c in ("env_amp_bpfo", "env_amp_bpfi", "env_amp_bsf"))
    return float(worst * margin)


# ---------------------------------------------------------------- plots
def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{title}\naccuracy = {accuracy_score(y_true, y_pred):.1%}")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def main():
    os.makedirs(PLOTS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    # --- step 1 + 2 -------------------------------------------------------
    data, defects = {}, {}
    print("=" * 68 + "\nSTEP 1-2: loading + bearing kinematics\n" + "=" * 68)
    for cls, (fname, sig_key, rpm_key) in FILES.items():
        sig, fr = load_class(fname, sig_key, rpm_key)
        windows = segment(sig)
        data[cls] = (sig, fr, windows)
        defects[cls] = bearing_defect_frequencies(fr, N_BALLS, D_BALL, D_PITCH, CONTACT_ANGLE)
        print(f"{cls:<11} {len(sig):>7} samples  {len(windows):>4} windows  "
              f"{fr * 60:.0f} rpm (fr = {fr:.2f} Hz)")

    table = pd.DataFrame([
        {"class": c, "rpm": data[c][1] * 60, "fr_hz": data[c][1], **defects[c]}
        for c in CLASSES
    ]).round(2)
    table.to_csv(f"{RESULTS}/defect_frequency_table.csv", index=False)
    print("\nTheoretical defect frequencies (Hz) - ground truth, not fitted:")
    print(table.to_string(index=False))

    # --- step 3: raw FFT, the plot that fails ------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for ax, cls in zip(axes, CLASSES):
        _, fr, windows = data[cls]
        freqs, _ = raw_spectrum(windows[0])
        mag = np.mean([raw_spectrum(w)[1] for w in windows], axis=0)
        keep = freqs <= 1000
        ax.plot(freqs[keep], mag[keep], lw=0.7, color="0.25")
        for k in (1, 2, 3):
            ax.axvline(k * fr, color="tab:blue", ls=":", lw=1)
        for name, color in (("BPFO", "tab:red"), ("BPFI", "tab:green"), ("BSF", "tab:orange")):
            ax.axvline(defects[cls][name], color=color, ls="--", lw=1.2, label=name)
        ax.set_title(f"{cls} - raw FFT (dotted blue = 1x/2x/3x shaft speed)", fontsize=10)
        ax.set_ylabel("|A| (g)")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Step 3: the raw spectrum does NOT reveal the defect frequency\n"
                 "energy sits at running speed and broadband noise - the dashed "
                 "defect lines have no distinct peak", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{PLOTS}/raw_fft_comparison.png", dpi=150)
    plt.close(fig)

    # --- step 4: envelope spectrum, the money plot -------------------------
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    for ax, cls in zip(axes, CLASSES):
        windows = data[cls][2]
        specs = [envelope_spectrum(w) for w in windows[:20]]
        freqs = specs[0][0]
        mag = np.mean([s[1] for s in specs], axis=0)
        ax.plot(freqs, mag, lw=0.8, color="0.2")
        for name, color in (("BPFO", "tab:red"), ("BPFI", "tab:green"), ("BSF", "tab:orange")):
            f = defects[cls][name]
            ax.axvline(f, color=color, ls="--", lw=1.3, label=f"{name} = {f:.1f} Hz")
        ax.set_title(f"{cls} - envelope spectrum ({BAND[0]}-{BAND[1]} Hz band)", fontsize=10)
        ax.set_ylabel("|A| (g)")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Step 4: envelope spectrum - peaks land on the theoretical "
                 "defect frequencies and their harmonics", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"{PLOTS}/envelope_spectrum_diagnosis.png", dpi=150)
    plt.close(fig)

    # --- step 5: features --------------------------------------------------
    feat_cols = ["rms", "kurtosis", "crest_factor",
                 "env_amp_bpfo", "env_amp_bpfi", "env_amp_bsf"]
    df = pd.concat([extract_features(data[c][2], defects[c], c) for c in CLASSES],
                   ignore_index=True)
    df.to_csv(f"{RESULTS}/feature_dataset.csv", index=False)
    print(f"\n{'=' * 68}\nSTEP 5: feature dataset - {df.shape[0]} windows x 6 features\n{'=' * 68}")
    print(df.groupby("label")[feat_cols].mean().round(4).to_string())

    # --- step 6: rule-based ------------------------------------------------
    print(f"\n{'=' * 68}\nSTEP 6: rule-based diagnosis (zero training)\n{'=' * 68}")
    nominal = accuracy_score(df["label"], df.apply(diagnose_fault, axis=1))
    thr = calibrate_threshold(df)
    df["rule_pred"] = df.apply(diagnose_fault, axis=1, noise_floor_threshold=thr)
    acc = accuracy_score(df["label"], df["rule_pred"])
    print(f"threshold 3.00 (nominal)              -> accuracy = {nominal:.1%}"
          "   (false alarms on healthy bearings)")
    print(f"threshold {thr:.2f} (healthy-calibrated)  -> accuracy = {acc:.1%}")
    cm = plot_confusion(df["label"], df["rule_pred"],
                        f"Rule-based (physics) diagnosis, threshold = {thr:.2f}",
                        f"{PLOTS}/confusion_matrix_rulebased.png")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

    # --- step 6b: kurtogram-selected band, as a comparison ------------------
    print(f"\n{'=' * 68}\nSTEP 6b: kurtogram band selection (Antoni 2006)\n{'=' * 68}")
    kbands = {}
    for cls in CLASSES:
        # one band per RECORD, not per class label - a real machine gives you one
        # record of unknown state, so this selection uses no labels.
        record = data[cls][0][:FS * 4]      # 4 s is enough, and far steadier
        kmap = kurtogram(record)             # than one window for band selection
        kbands[cls] = select_band(record)
        (lo, hi), k = kbands[cls]
        print(f"{cls:<11} selected {lo:>6.0f}-{hi:<6.0f} Hz   max spectral kurtosis = {k:.2f}")
        if cls in ("Ball", "Outer Race"):   # outer race is the contrast case
            tag = "ball_fault_B007" if cls == "Ball" else "outer_race_OR007"
            plot_kurtogram(kmap, kbands[cls], f"Kurtogram - {cls}",
                           f"{PLOTS}/kurtogram_{tag}.png")

    dfk = pd.concat([extract_features(data[c][2], defects[c], c, band=kbands[c][0])
                     for c in CLASSES], ignore_index=True)
    dfk.to_csv(f"{RESULTS}/feature_dataset_kurtogram.csv", index=False)
    thr_k = calibrate_threshold(dfk)
    dfk["rule_pred"] = dfk.apply(diagnose_fault, axis=1, noise_floor_threshold=thr_k)
    acc_k = accuracy_score(dfk["label"], dfk["rule_pred"])
    # same threshold as the fixed band, so the bands are compared and not the
    # calibrations; and each band's own calibration, which is the honest
    # end-to-end number.
    acc_k_same = accuracy_score(dfk["label"],
                                dfk.apply(diagnose_fault, axis=1, noise_floor_threshold=thr))
    print(f"\nfixed 2-5 kHz band                       -> accuracy = {acc:.1%}  (threshold {thr:.2f})")
    print(f"kurtogram band, own calibration          -> accuracy = {acc_k:.1%}  (threshold {thr_k:.2f})")
    print(f"kurtogram band, fixed-band threshold     -> accuracy = {acc_k_same:.1%}  (threshold {thr:.2f})")
    cm_k = plot_confusion(dfk["label"], dfk["rule_pred"],
                          f"Rule-based, kurtogram-selected band, threshold = {thr_k:.2f}",
                          f"{PLOTS}/confusion_matrix_rulebased_kurtogram.png")
    print(pd.DataFrame(cm_k, index=CLASSES, columns=CLASSES).to_string())

    print("\nDefect-line-to-noise ratio (median over windows), fixed vs kurtogram band:")
    print(f"{'class':<11} {'bandwidth':>10} {'BSF fix':>8} {'BSF kur':>8} "
          f"{'2xBSF kur':>10} {'own-line fix':>13} {'own-line kur':>13}")
    own = {"Normal": "env_amp_bsf", "Inner Race": "env_amp_bpfi",
           "Ball": "env_amp_bsf", "Outer Race": "env_amp_bpfo"}
    for cls in CLASSES:
        (lo, hi), _ = kbands[cls]
        f, k = df[df["label"] == cls], dfk[dfk["label"] == cls]
        r2 = np.median([peak_at(*envelope_spectrum(w, band=(lo, hi)),
                                2 * defects[cls]["BSF"]) / nb
                        for w, nb in zip(data[cls][2][:20], k["baseline_noise"][:20])])
        print(f"{cls:<11} {hi - lo:>9.0f}Hz "
              f"{np.median(f['env_amp_bsf'] / f['baseline_noise']):>8.1f} "
              f"{np.median(k['env_amp_bsf'] / k['baseline_noise']):>8.1f} "
              f"{r2:>10.1f} "
              f"{np.median(f[own[cls]] / f['baseline_noise']):>13.1f} "
              f"{np.median(k[own[cls]] / k['baseline_noise']):>13.1f}")

    # --- step 7: ML cross-check -------------------------------------------
    Xtr, Xte, ytr, yte = train_test_split(df[feat_cols], df["label"], test_size=0.3,
                                          stratify=df["label"], random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    pred = rf.predict(Xte)
    print(f"\n{'=' * 68}\nSTEP 7: RandomForest cross-check\n{'=' * 68}")
    print(f"accuracy = {accuracy_score(yte, pred):.1%}")
    cm_ml = plot_confusion(yte, pred, "RandomForest cross-check",
                           f"{PLOTS}/confusion_matrix_ml.png")
    print(pd.DataFrame(cm_ml, index=CLASSES, columns=CLASSES).to_string())

    imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values()
    print("\nFeature importances:")
    print(imp.iloc[::-1].round(4).to_string())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values,
            color=["tab:red" if f.startswith("env_amp") else "0.6" for f in imp.index])
    ax.set_xlabel("importance")
    ax.set_title("RandomForest feature importance\n(red = physics-derived envelope features)")
    fig.tight_layout()
    fig.savefig(f"{PLOTS}/feature_importance.png", dpi=150)
    plt.close(fig)
    print(f"\nArtifacts written to {PLOTS}/ and {RESULTS}/")


def _self_check():
    """assert-based sanity check on the physics and the envelope core."""
    f = bearing_defect_frequencies(29.95, N_BALLS, D_BALL, D_PITCH)
    assert abs(f["BPFO"] - 107.4) < 1.0, f["BPFO"]      # CWRU published 6205 values
    assert abs(f["BPFI"] - 162.2) < 1.0, f["BPFI"]
    # BSF here is the ball SPIN frequency; CWRU's published 141.2 Hz is 2x this,
    # because a ball defect strikes the inner and outer race once per revolution.
    assert abs(f["BSF"] - 70.6) < 1.0, f["BSF"]
    assert abs(f["BPFO"] + f["BPFI"] - N_BALLS * 29.95) < 1e-9   # identity: sum = n*fr

    # a 3 kHz carrier amplitude-modulated at 107 Hz must show 107 Hz in the envelope
    t = np.arange(FS) / FS
    sig = (1 + 0.5 * np.sign(np.sin(2 * np.pi * 107 * t))) * np.sin(2 * np.pi * 3000 * t)
    freqs, mag = envelope_spectrum(sig)
    assert abs(freqs[np.argmax(mag)] - 107) < 2, freqs[np.argmax(mag)]
    assert peak_at(freqs, mag, 107) > 10 * baseline_noise(freqs, mag, [107])

    assert segment(np.zeros(24000), 12000, 12000).shape == (2, 12000)

    # the rule must fire on a strong BPFO line and stay quiet on a flat spectrum
    loud = {"env_amp_bpfo": 1.0, "env_amp_bpfi": 0.01, "env_amp_bsf": 0.01,
            "baseline_noise": 0.01}
    assert diagnose_fault(loud) == "Outer Race"
    assert diagnose_fault(dict(loud, env_amp_bpfo=0.02)) == "Normal"

    # the kurtogram must find a resonance it was not told about: impulses at
    # 40 Hz ringing a 2.5 kHz resonance under wideband noise. (The selection is
    # SNR-limited by nature - at noise amplitude 0.5 it no longer resolves it.)
    rng = np.random.default_rng(0)
    ring, n = np.zeros(FS * 2), 200         # ring must decay before the next
    decay = np.arange(n)                    # impact, or it is not impulsive
    decay = np.exp(-decay / 40) * np.sin(2 * np.pi * 2500 * decay / FS)
    for start in range(0, len(ring) - n, int(FS / 40)):
        ring[start:start + n] += decay
    (lo, hi), _ = select_band(ring + 0.1 * rng.standard_normal(len(ring)))
    assert lo <= 2500 <= hi, (lo, hi)

    print("self-check OK")


if __name__ == "__main__":
    _self_check()
    main()
