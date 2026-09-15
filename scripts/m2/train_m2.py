"""
M2 -- Telemetry LSTM (CAN-only baseline).

Requires scripts/common/preprocess_m1_m2_m3.py to have been run first
(datasets/processed/ul_dd_regen/fold_{0..4}.npz must exist).

Architecture, training loop, and metrics are all Kavipriya's own unmodified code
(src/models/architecture.py:build_m2_telemetry_lstm, src/models/train.py:cross_validate).
"""
import sys
import json
from pathlib import Path

REPO = "/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, REPO)

from models.architecture import build_m2_telemetry_lstm
from models.train import cross_validate

FOLD_DIR = f"{REPO}/datasets/processed/ul_dd_regen"
CKPT_DIR = f"{REPO}/models/checkpoints_regen"
RESULTS_DIR = f"{REPO}/results/reports_regen"


def main():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    print("\n########## M2 ##########")
    result = cross_validate(
        model_fn=build_m2_telemetry_lstm, model_name='M2', fold_dir=FOLD_DIR,
        input_keys=('mm_tele_train', 'mm_tele_test'), label_keys=('mm_y_train', 'mm_y_test'),
        save_dir=CKPT_DIR,
    )
    json.dump(result, open(f"{RESULTS_DIR}/M2_results.json", 'w'), default=str, indent=2)
    print(f"M2: acc={result['mean_acc']:.4f}+-{result['std_acc']:.4f}  "
          f"macro_f1={result['mean_f1']:.4f}+-{result['std_f1']:.4f}")


if __name__ == "__main__":
    main()
