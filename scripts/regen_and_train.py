import sys, json, time
REPO = "/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system"
sys.path.insert(0, f"{REPO}/src")
sys.path.insert(0, REPO)

from pathlib import Path
from data.preprocess import ULDDProcessor

DATASET_ROOT = "/Users/karthikeyansundaravaradhan/Desktop/Masters Project/DriverDrowsinessDataset"
OUT_DIR = f"{REPO}/datasets/processed/ul_dd_regen"

proc = ULDDProcessor(dataset_root=DATASET_ROOT, processed_dir=OUT_DIR)
# Fix the triple-nested path assumption in the original code (our copy is double-nested)
proc.fau_root = proc.root / 'Extracted_Features' / 'Extracted_Features'
proc.tele_root = proc.root / 'CSV_Files' / 'CSV_Files'
print("fau_root  =", proc.fau_root, "exists:", proc.fau_root.exists())
print("tele_root =", proc.tele_root, "exists:", proc.tele_root.exists())
print("labels    =", proc.labels_path, "exists:", proc.labels_path.exists())

t0 = time.time()
folds = proc.run(verbose=True)
print(f"\nPreprocessing took {time.time()-t0:.1f}s")

# ── Now train M1, M2, M3 fresh on this regenerated, internally-consistent data ──
import tensorflow as tf
from models.architecture import build_m1_facial_bilstm, build_m2_telemetry_lstm, M3FusionModel
from models.train import cross_validate

CKPT_DIR = f"{REPO}/models/checkpoints_regen"
RESULTS_DIR = f"{REPO}/results/reports_regen"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

all_summaries = {}

print("\n\n########## M1 ##########")
r1 = cross_validate(
    model_fn=build_m1_facial_bilstm, model_name='M1', fold_dir=OUT_DIR,
    input_keys=('X_fau_train', 'X_fau_test'), label_keys=('y_train', 'y_test'),
    save_dir=CKPT_DIR,
)
all_summaries['M1'] = {k: v for k, v in r1.items() if k != 'folds'}
json.dump(r1, open(f"{RESULTS_DIR}/M1_results.json", 'w'), default=str, indent=2)

print("\n\n########## M2 ##########")
r2 = cross_validate(
    model_fn=build_m2_telemetry_lstm, model_name='M2', fold_dir=OUT_DIR,
    input_keys=('mm_tele_train', 'mm_tele_test'), label_keys=('mm_y_train', 'mm_y_test'),
    save_dir=CKPT_DIR,
)
all_summaries['M2'] = {k: v for k, v in r2.items() if k != 'folds'}
json.dump(r2, open(f"{RESULTS_DIR}/M2_results.json", 'w'), default=str, indent=2)

print("\n\n########## M3 ##########")
r3 = cross_validate(
    model_fn=lambda: M3FusionModel(), model_name='M3', fold_dir=OUT_DIR,
    input_keys=(['mm_fau_train', 'mm_tele_train'], ['mm_fau_test', 'mm_tele_test']),
    label_keys=('mm_y_train', 'mm_y_test'),
    save_dir=CKPT_DIR,
)
all_summaries['M3'] = {k: v for k, v in r3.items() if k != 'folds'}
json.dump(r3, open(f"{RESULTS_DIR}/M3_results.json", 'w'), default=str, indent=2)

print("\n\n========== FINAL SUMMARY ==========")
for name, s in all_summaries.items():
    print(f"{name}: acc={s.get('mean_acc', 'NA')} +- {s.get('std_acc','NA')}   macro_f1={s.get('mean_f1','NA')} +- {s.get('std_f1','NA')}")
