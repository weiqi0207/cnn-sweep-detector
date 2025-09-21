#!/usr/bin/env python3
import os, glob, json, math, random, argparse, re
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torchvision.transforms import InterpolationMode

# --------------------------
# Args
# --------------------------
def get_args():
    p = argparse.ArgumentParser(description="Train sweep vs non-sweep CNN on ms_png images with SWA.")
    p.add_argument("--root",
        default="/work/users/w/a/wang0207/GHIST/Data Files and Submission Templates/ss_by_CNN/ms_png",
        type=str, help="Root folder containing PNGs (recursive).")
    p.add_argument("--val-frac",    default=0.10, type=float, help="Fraction for validation split.")
    p.add_argument("--epochs",      default=12,   type=int,   help="Training epochs.")
    p.add_argument("--batch-size",  default=32,   type=int,   help="Batch size.")
    p.add_argument("--lr",          default=1e-3, type=float, help="Base LR for Adam.")
    p.add_argument("--swa-start-frac", default=0.7, type=float, help="When to start SWA (fraction of epochs).")
    p.add_argument("--swa-lr",      default=5e-4, type=float, help="LR during SWA phase.")
    p.add_argument("--num-workers", default=int(os.environ.get("NUM_WORKERS", "8")), type=int, help="Dataloader workers.")
    p.add_argument("--seed",        default=int(os.environ.get("SEED", "42")), type=int, help="Random seed.")
    # Your ms→PNG pipeline produces (H=haplotypes≈50, W=512); keep that as default to avoid distortion.
    p.add_argument("--resize",      default="50x512", type=str,
                   help="HxW resize (e.g., 50x512). Use NEAREST to preserve binary look.")
    return p.parse_args()

# --------------------------
# Reproducibility
# --------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------
# Data
# --------------------------
class PngDataset(Dataset):
    """Expects columns: filepath (str), label (0/1)."""
    def __init__(self, df: pd.DataFrame, tfm: T.Compose):
        self.paths  = df["filepath"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tfm    = tfm

    def __len__(self): return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("L")
        x   = self.tfm(img)
        y   = torch.tensor(self.labels[i], dtype=torch.long)
        if not x.is_contiguous(): x = x.contiguous()
        return x, y

def build_filelist(root: str):
    pngs = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
    rows = []
    for p in pngs:
        base = os.path.basename(p)
        if re.search(r'_SWEEP\.png$', base):
            y = 1
        elif re.search(r'_NOSWEEP\.png$', base):
            y = 0
        else:
            continue
        rows.append({"filepath": p, "label": y})
    if not rows:
        raise SystemExit(f"No labeled PNGs found under {root} (need *_SWEEP.png or *_NOSWEEP.png).")
    df = pd.DataFrame(rows)
    vc = df["label"].value_counts()
    print("Class counts:", vc.to_dict())
    if df["label"].nunique() < 2:
        raise SystemExit("Only one class found; check filenames/labels.")
    return df

def compute_norm_stats(paths, max_imgs=512):
    """Mean/std over a subset for speed, clamp std to avoid divide-by-zero."""
    to_tensor = T.ToTensor()
    m = 0.0; s = 0.0; n = 0
    for p in random.sample(paths, k=min(len(paths), max_imgs)):
        x = to_tensor(Image.open(p).convert("L"))  # (1,H,W) in [0,1]
        m += x.mean().item()
        s += x.std(unbiased=False).item()
        n += 1
    n = max(1, n)
    mean = [m / n]
    std  = [max(s / n, 1e-6)]
    return mean, std

def make_transforms(mean, std, resize_spec: str):
    try:
        H, W = map(int, resize_spec.lower().split("x"))
    except Exception:
        raise SystemExit(f"Bad --resize '{resize_spec}', expected like 50x512.")
    ops = [
        T.Resize((H, W), interpolation=InterpolationMode.NEAREST),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]
    return T.Compose(ops)

# --------------------------
# Model
# --------------------------
def build_model():
    # 1×50×512 input (default). First pool (2,1) reduces height faster than width.
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d((2, 1)),  # -> 16×25×512
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # -> 32×12×256
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),      # -> 64×6×128
        nn.Flatten(),
        nn.LazyLinear(128), nn.ReLU(), nn.Dropout(0.5),
        nn.LazyLinear(2)
    )

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    n_correct = 0; n_total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        pred = model(xb).argmax(1)
        n_correct += (pred == yb).sum().item()
        n_total   += yb.numel()
    return (n_correct / max(1, n_total)) if n_total else float("nan")

# --------------------------
# Main
# --------------------------
def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = build_filelist(args.root)

    # Stratified split (fallback if class too small)
    label_counts = df["label"].value_counts()
    if df["label"].nunique() < 2 or label_counts.min() < 2:
        train_df, val_df = train_test_split(df, test_size=args.val_frac, random_state=args.seed, shuffle=True)
    else:
        train_df, val_df = train_test_split(df, test_size=args.val_frac, stratify=df["label"], random_state=args.seed)

    print(f"Train={len(train_df)}  Val={len(val_df)}  "
          f"pos_rate_train={train_df.label.mean():.3f}  pos_rate_val={val_df.label.mean():.3f}")

    # Normalize on TRAIN only
    mean, std = compute_norm_stats(train_df["filepath"].tolist())
    print("Normalize mean/std:", mean, std)
    tfm = make_transforms(mean, std, args.resize)

    train_dl = DataLoader(PngDataset(train_df, tfm), batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True if args.num_workers>0 else False)
    val_dl   = DataLoader(PngDataset(val_df,   tfm), batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True, persistent_workers=True if args.num_workers>0 else False)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Warm up LazyLinear
    if len(train_dl.dataset):
        xb0, _ = next(iter(train_dl)); xb0 = xb0.to(device)[:1]
        with torch.no_grad(): _ = model(xb0)

    # SWA
    swa_model   = AveragedModel(model)
    swa_start_e = int(math.ceil(args.swa_start_frac * args.epochs))
    swa_sched   = SWALR(optimizer, swa_lr=args.swa_lr)

    best_val = -1.0
    run_id   = random.randint(100000, 999999)
    base     = f"cnn_ms_sweep_SWA_{run_id}"
    best_path = f"{base}.best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0; correct = 0; total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total   += yb.size(0)

        if epoch >= swa_start_e:
            swa_model.update_parameters(model)
            swa_sched.step()

        train_loss = running_loss / max(1, total)
        train_acc  = correct / max(1, total)
        val_acc    = eval_acc(model, val_dl, device) if len(val_dl.dataset) else float("nan")
        print(f"Epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if not math.isnan(val_acc) and val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"[BEST] val_acc={val_acc:.4f} saved -> {best_path}")

    # Finalize SWA
    if len(train_dl.dataset):
        try:
            update_bn(train_dl, swa_model, device=device)
        except Exception as e:
            print(f"[WARN] update_bn skipped: {e}")

    torch.save(swa_model.state_dict(), best_path)
    print(f"[OK] Saved SWA weights -> {best_path}")

    meta = {"mean": mean, "std": std, "class_map": {"NOSWEEP": 0, "SWEEP": 1},
            "seed": args.seed, "args": vars(args)}
    with open(f"{base}.meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Saved meta -> {base}.meta.json")

if __name__ == "__main__":
    main()