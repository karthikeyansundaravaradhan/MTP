"""C9 runner: VIGIL gate vs vision-only / ungated / clock control, under vision blackouts.

  python scripts/m8/run_vigil.py                               # V* = ir_112, all four modes
  python scripts/m8/run_vigil.py --vision ir_112 pose_224 --name C9_ir_pose
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m8.vigil import default_cfg, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", nargs="+", default=["ir_112"])
    ap.add_argument("--modes", nargs="+", default=["vision", "gate", "late", "clock"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gate-lr-mult", type=float, default=1.0, help="gate learning-rate multiplier (no wd on gate)")
    ap.add_argument("--y", default="y3", choices=["y3", "y_b2"], help="B2: KSS <= 5 vs >= 6")
    ap.add_argument("--name")
    args = ap.parse_args()
    cfg = default_cfg(vision=args.vision, modes=args.modes, seeds=args.seeds, epochs=args.epochs,
                      gate_lr_mult=args.gate_lr_mult, y_col=args.y, n_classes=3 if args.y == "y3" else 2)
    run(args.name or "C9_" + "_".join(args.vision), cfg)


if __name__ == "__main__":
    main()
