#!/usr/bin/env python3
import os, re, json, argparse, glob
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# ---- Model (must match training) ----
def build_model():
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d((2, 1)),
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.LazyLinear(128), nn.ReLU(), nn.Dropout(0.5),
        nn.LazyLinear(2)
    )

# ---- Utils ----
def parse_resize(spec: str):
    try:
        H, W = map(int, spec.lower().split("x"))
        return H, W
    except Exception:
        raise SystemExit(f"Bad resize spec in meta: '{spec}' (expected like 50x512)")

def make_transforms(mean, std, resize_hw):
    H, W = resize_hw
    return T.Compose([
        T.Resize((H, W), interpolation=InterpolationMode.NEAREST),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

def guess_window_from_name(path: Path):
    m = re.search(r"chr(\w+)_([0-9]+)_([0-9]+)", path.name, flags=re.IGNORECASE)
    if not m:
        return ("NA", None, None)
    chrom = m.group(1)
    lo = int(m.group(2)); hi = int(m.group(3))
    return (chrom, lo, hi)

# ---- Main ----
def get_args():
    ap = argparse.ArgumentParser(description="Predict SWEEP vs NOSWEEP on PNGs using trained CNN.")
    ap.add_argument("--imgs",
        default="vcf_png", help="Folder with PNGs from vcf_to_png_fixed.py")
    ap.add_argument("--ckpt", required=True, help="Path to *.best.pt checkpoint")
    ap.add_argument("--meta", default="", help="Path to *.meta.json (if empty, infer from --ckpt)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS","4")))
    ap.add_argument("--out", default="predictions.csv", help="Output CSV")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    ckpt_path = Path(args.ckpt)
    meta_path = Path(args.meta) if args.meta else Path(str(ckpt_path).replace(".best.pt", ".meta.json"))
    if not meta_path.exists():
        raise SystemExit(f"Meta JSON not found: {meta_path}")

    # Load meta (mean/std, resize, etc.)
    meta = json.loads(Path(meta_path).read_text())
    mean = meta.get("mean", [0.5])
    std  = meta.get("std", [0.5])
    resize_spec = meta.get("args", {}).get("resize", "50x512")
    resize_hw = parse_resize(resize_spec)

    # Build model + warmup LazyLinear
    device = torch.device(args.device)
    model = build_model().to(device)
    # Warm-up with a fake sample of correct size
    _ = model(torch.zeros(1, 1, resize_hw[0], resize_hw[1], device=device))

    # Load weights
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Transforms
    tfm = make_transforms(mean, std, resize_hw)

    # Collect images
    img_paths = sorted(glob.glob(os.path.join(args.imgs, "**", "*.png"), recursive=True))
    if not img_paths:
        raise SystemExit(f"No PNGs found under {args.imgs}")

    # Inference
    rows = []
    softmax = nn.Softmax(dim=1)
    batch, tensors, metas = [], [], []
    def flush():
        nonlocal tensors, metas, rows
        if not tensors: return
        xb = torch.stack(tensors, dim=0).to(device)           # (B,1,H,W)
        logits = model(xb)                                    # (B,2)
        probs = softmax(logits).detach().cpu().numpy()        # [:,0]=NOSWEEP, [:,1]=SWEEP
        for m, pr in zip(metas, probs):
            pred = int(pr[1] >= pr[0])                        # 1=SWEEP, 0=NOSWEEP
            rows.append({
                "filepath": m["path"],
                "chrom": m["chrom"],
                "win_lo": m["win_lo"],
                "win_hi": m["win_hi"],
                "prob_NOSWEEP": float(pr[0]),
                "prob_SWEEP": float(pr[1]),
                "pred_label": "SWEEP" if pred==1 else "NOSWEEP",
                "pred_int": pred,
            })
        tensors, metas = [], []

    for p in img_paths:
        img = Image.open(p).convert("L")
        x = tfm(img)  # (1,H,W)
        chrom, lo, hi = guess_window_from_name(Path(p))
        tensors.append(x)
        metas.append({"path": p, "chrom": chrom, "win_lo": lo, "win_hi": hi})
        if len(tensors) >= args.batch-size:
            flush()
    flush()

    # Save CSV
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {len(df)} predictions")

if __name__ == "__main__":
    main()
