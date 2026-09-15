# ARCHIVED -- superseded. This was an early attempt to evaluate Kavipriya's *original*
# checkpoints (which turned out to be Git LFS pointer stubs, not real weights -- see
# docs/repo_understanding.md section 8) against datasets/processed/ul_dd. The project
# retrained from scratch instead (scripts/common/preprocess_m1_m2_m3.py + scripts/m1-3/).
# Kept only for history, not needed going forward.
import sys, json
sys.path.insert(0, "/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system/src")
sys.path.insert(0, "/Users/karthikeyansundaravaradhan/Desktop/Masters Project/driver-drowsiness-detection-system")

import numpy as np
from inference.predict import load_predictor_from_fold

FOLD_DIR = "datasets/processed/ul_dd"
CKPT_DIR = "models/checkpoints"

results = {}
for model_name, n_folds in [("M1", 5), ("M2", 5), ("M3", 1)]:
    accs = []
    f1s = []
    for fold_idx in range(n_folds):
        try:
            pred = load_predictor_from_fold(FOLD_DIR, fold_idx, model_name, CKPT_DIR)
        except Exception as e:
            print(f"{model_name} fold{fold_idx}: LOAD FAILED: {e}")
            continue
        m = pred.evaluate_fold(f"{FOLD_DIR}/fold_{fold_idx}.npz", fold_idx)
        accs.append(m["accuracy"])
        f1s.append(m["macro_f1"])
        print(f"{model_name} fold{fold_idx}: acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    if accs:
        results[model_name] = {
            "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
            "mean_f1": float(np.mean(f1s)), "std_f1": float(np.std(f1s)),
            "n_folds": len(accs),
        }
        print(f"{model_name} OVERALL: acc={np.mean(accs):.4f}+-{np.std(accs):.4f}  macro_f1={np.mean(f1s):.4f}+-{np.std(f1s):.4f}")
    print()

print(json.dumps(results, indent=2))
