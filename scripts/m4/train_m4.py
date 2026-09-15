import sys, json
from pathlib import Path

REPO = "/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, REPO)

from models.architecture import build_m4_realtime_bilstm
from models.train import cross_validate

FOLD_DIR = f"{REPO}/datasets/processed/uldd_m4"
CKPT_DIR = f"{REPO}/models/checkpoints_regen"
RESULTS_DIR = f"{REPO}/results/reports_regen"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

print("\n\n########## M4 ##########")
r = cross_validate(
    model_fn=build_m4_realtime_bilstm, model_name='M4', fold_dir=FOLD_DIR,
    input_keys=('X_m4_train', 'X_m4_test'), label_keys=('y_train', 'y_test'),
    save_dir=CKPT_DIR,
)
json.dump(r, open(f"{RESULTS_DIR}/M4_results.json", 'w'), default=str, indent=2)
print(f"\nM4: acc={r.get('mean_acc','NA')} +- {r.get('std_acc','NA')}  "
      f"macro_f1={r.get('mean_f1','NA')} +- {r.get('std_f1','NA')}")
