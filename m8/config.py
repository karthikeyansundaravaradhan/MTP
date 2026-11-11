from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "DriverDrowsinessDataset"
VIDEO_ROOT = DATA / "Video_Data" / "Video_Data"
FEAT_ROOT = DATA / "Extracted_Features" / "Extracted_Features"
CSV_ROOT = DATA / "CSV_Files" / "CSV_Files"
LABELS_CSV = DATA / "Labels.csv"
KAVI = ROOT / "driver-drowsiness-detection-system"

CACHE = ROOT / "cache"
RESULTS = ROOT / "results" / "m8"

SUBJECTS = list("ABCDEFGHIJKLMNOPQRS")
FOLDS = [list("ABCD"), list("EFGH"), list("IJK"), list("LMNO"), list("PQRS")]
NO_VIDEO = frozenset("BIM")
NO_TELEMETRY = frozenset("A")
AWAKE_ONLY = frozenset("CFL")

FPS = 60
SESSION_SEC = 2400
SESSION_FRAMES = FPS * SESSION_SEC
EPOCH_SEC = 240
WINDOW_SEC = 60
STRIDE_SEC = 15
N_INNER = 4

CLASS_NAMES = ["Alert", "LowVigilant", "Drowsy"]


def kss_to_y3(kss):
    return 0 if kss <= 3 else (1 if kss <= 6 else 2)


def kss_to_b2(kss):
    return 0 if kss <= 5 else 1


def outer_fold_of(subject):
    return next(i for i, f in enumerate(FOLDS) if subject in f)
