"""
M3 -- Cross-modal attention fusion (FAU + Telemetry). Thesis's main contribution.

Requires scripts/common/preprocess_m1_m2_m3.py to have been run first
(datasets/processed/ul_dd_regen/fold_{0..4}.npz must exist).

Architecture, training loop, and metrics are all Kavipriya's own unmodified code
(src/models/architecture.py:M3FusionModel, src/models/train.py:cross_validate).
"""
import sys
import json
from pathlib import Path

REPO = "/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, REPO)

from models.architecture import M3FusionModel
from models.train import cross_validate

FOLD_DIR = f"{REPO}/datasets/processed/ul_dd_regen"
CKPT_DIR = f"{REPO}/models/checkpoints_regen"
RESULTS_DIR = f"{REPO}/results/reports_regen"


def main():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    print("\n########## M3 ##########")
    result = cross_validate(
        model_fn=lambda: M3FusionModel(), model_name='M3', fold_dir=FOLD_DIR,
        input_keys=(['mm_fau_train', 'mm_tele_train'], ['mm_fau_test', 'mm_tele_test']),
        label_keys=('mm_y_train', 'mm_y_test'),
        save_dir=CKPT_DIR,
    )
    json.dump(result, open(f"{RESULTS_DIR}/M3_results.json", 'w'), default=str, indent=2)
    print(f"M3: acc={result['mean_acc']:.4f}+-{result['std_acc']:.4f}  "
          f"macro_f1={result['mean_f1']:.4f}+-{result['std_f1']:.4f}")


if __name__ == "__main__":
    main()
