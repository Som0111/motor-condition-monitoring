"""
Step 0 — Run this BEFORE Claude Code.

CWRU BLOCKS SCRIPTED DOWNLOADS (403 Forbidden) - you must download manually
via browser, then run this script to inspect the .mat key names.

MANUAL DOWNLOAD STEPS:
1. Go to: https://engineering.case.edu/bearingdatacenter/download-data-file
2. Under "12k Drive End Bearing Fault Data", 1797 rpm (0 HP load) row, download:
   - Normal baseline: from the "Normal Baseline Data" table, 1797 rpm row -> save as normal.mat
   - Inner Race 0.007": IR007, 1797rpm row -> save as inner_race.mat
   - Ball 0.007": B007, 1797rpm row -> save as ball.mat
   - Outer Race 0.007" (Centered, @6:00): OR007@6, 1797rpm row -> save as outer_race.mat
3. Create a folder "cwru_data" next to this script and put all 4 files in it.
4. Run: python3 download_and_inspect.py
"""

import scipy.io
import os

SAVE_DIR = "cwru_data"
os.makedirs(SAVE_DIR, exist_ok=True)

FILES = ["normal.mat", "inner_race.mat", "ball.mat", "outer_race.mat"]

missing = [f for f in FILES if not os.path.exists(os.path.join(SAVE_DIR, f))]
if missing:
    print(f"Missing files in {SAVE_DIR}/: {missing}")
    print("Download them manually first (see instructions in this script's docstring).\n")

print("--- Inspecting .mat file keys ---\n")
for fname in FILES:
    path = os.path.join(SAVE_DIR, fname)
    if not os.path.exists(path):
        print(f"{fname}: NOT FOUND, skipped\n")
        continue
    try:
        data = scipy.io.loadmat(path)
        keys = [k for k in data.keys() if not k.startswith("__")]
        print(f"{fname}:")
        for k in keys:
            shape = data[k].shape if hasattr(data[k], "shape") else "?"
            print(f"    key='{k}'  shape={shape}")
        de_keys = [k for k in keys if "DE_time" in k]
        print(f"    -> Drive-End channel key: {de_keys}\n")
    except Exception as e:
        print(f"{fname}: ERROR reading file - {e}\n")

print("Done. Give Claude Code the SAVE_DIR path and the printed DE_time key names.")
