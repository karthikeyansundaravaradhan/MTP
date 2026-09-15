"""
Shared preprocessing for M1/M2/M3 -- regenerates the UL-DD dataset fold files
(datasets/processed/ul_dd_regen/fold_{0..4}.npz) that scripts/m1/train_m1.py,
scripts/m2/train_m2.py, and scripts/m3/train_m3.py all read from.

Run this once before training any of M1/M2/M3 (or whenever the raw dataset changes).
Uses Kavipriya's own ULDDProcessor unmodified, only overriding fau_root/tele_root
to match this project's actual double-nested dataset layout (her code assumes
triple-nested -- see docs/repo_understanding.md section 6 for why).
"""
import sys
import time

REPO = "/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, REPO)

from data.preprocess import ULDDProcessor

DATASET_ROOT = "/Users/karthikeyansundaravaradhan/Desktop/Masters Project/DriverDrowsinessDataset"
OUT_DIR = f"{REPO}/datasets/processed/ul_dd_regen"


def main():
    proc = ULDDProcessor(dataset_root=DATASET_ROOT, processed_dir=OUT_DIR)
    # Fix the triple-nested path assumption in the original code (our copy is double-nested)
    proc.fau_root = proc.root / 'Extracted_Features' / 'Extracted_Features'
    proc.tele_root = proc.root / 'CSV_Files' / 'CSV_Files'
    print("fau_root  =", proc.fau_root, "exists:", proc.fau_root.exists())
    print("tele_root =", proc.tele_root, "exists:", proc.tele_root.exists())
    print("labels    =", proc.labels_path, "exists:", proc.labels_path.exists())

    t0 = time.time()
    proc.run(verbose=True)
    print(f"\nPreprocessing took {time.time()-t0:.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
