"""C1-C4 single-stream runs on cached features.

Examples:
  --stream ir_168            C1 (IR face crop, DINOv2-S)
  --stream ir_112            C1 resolution variant
  --stream l3d_168           C2 (stereo-left face)
  --stream pose_224          C3 (side view, 1 fps)
  --stream tab               C4 (landmarks/AU/pose geometry, 2 Hz)
  --head mean                linear-probe control for any stream
  --stream l3d_168 --tta-stream r3d_168   C2 two-view training + test-time averaging
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m8.data import series_bank, video_bank  # noqa: E402
from m8.train import default_cfg, run_cv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--head", default="mil")
    ap.add_argument("--tokens", default="cls_mean", help="video token mode: cls_mean | cls | all")
    ap.add_argument("--cohort", default="video")
    ap.add_argument("--y", default="y3", choices=["y3", "y_b2"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-inner", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--context", type=int, default=60, help="causal look-back in seconds (video streams)")
    ap.add_argument("--loss", default="ce", choices=["ce", "ordinal"], help="C10: ordinal cumulative-logit loss")
    ap.add_argument("--ref", type=int, default=0, help="C12: subtract mean of the session's first N seconds")
    ap.add_argument("--ref-session", default="same", choices=["same", "A"],
                    help="R2-7: take the reference from the subject's Alert session")
    ap.add_argument("--ref-concat", action="store_true", help="R2-8: [absolute, absolute - reference]")
    ap.add_argument("--ctx-stride", type=int, default=1, help="keep every k-th frame of the context")
    ap.add_argument("--tta-stream", help="second camera view of the same feature space (e.g. r3d_168)")
    ap.add_argument("--protocol", default="5fold", choices=["5fold", "loso"], help="V1: LOSO (implies --no-inner)")
    ap.add_argument("--name")
    args = ap.parse_args()

    if args.stream in ("tab", "tel"):
        make = lambda w: [series_bank(w, args.stream)]  # noqa: E731
    else:
        make = lambda w: [video_bank(w, args.stream, args.tokens, args.context, args.ctx_stride, args.ref,
                                   args.ref_session, args.ref_concat)]  # noqa: E731
    if args.tta_stream:
        make = lambda w: [[video_bank(w, args.stream, args.tokens)],  # noqa: E731
                          [video_bank(w, args.tta_stream, args.tokens)]]
    cfg = default_cfg(head=args.head, cohort=args.cohort, y_col=args.y,
                      n_classes=3 if args.y == "y3" else 2, seeds=args.seeds, epochs=args.epochs,
                      inner=not args.no_inner, stream=args.stream, tokens=args.tokens,
                      tta_stream=args.tta_stream, context=args.context,
                      ctx_stride=args.ctx_stride, ref=args.ref, loss=args.loss,
                      ref_session=args.ref_session, ref_concat=args.ref_concat, protocol=args.protocol)
    if args.protocol == "loso":
        cfg["inner"] = False
    name = args.name or (f"{args.stream}_{args.head}_{args.tokens}_{args.cohort}_{args.y}"
                         + (f"_tta-{args.tta_stream}" if args.tta_stream else "")
                         + (f"_ctx{args.context}" if args.context != 60 else "")
                         + (f"s{args.ctx_stride}" if args.ctx_stride != 1 else "")
                         + (f"_ref{args.ref}" if args.ref else "")
                         + ("A" if args.ref and args.ref_session == "A" else "")
                         + ("cat" if args.ref and args.ref_concat else "")
                         + ("_ord" if args.loss == "ordinal" else "")
                         + ("_loso" if args.protocol == "loso" else ""))
    run_cv(name, make, cfg, log_wandb=not args.no_wandb)


if __name__ == "__main__":
    main()
