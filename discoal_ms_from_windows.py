#!/usr/bin/env python3
import argparse, os, random, subprocess, sys
import numpy as np
import pandas as pd

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def build_windows(chrom: str, start: int, end: int, winsize: int, nrows: int):
    """
    Build window table with columns chrom, win_lo, win_hi, L_bp for exactly nrows windows
    spanning [start, end] in 1-based inclusive coordinates, with non-overlapping 100kb bins.
    """
    edges = list(range(start, end + 1, winsize))
    if edges[-1] <= end:
        edges.append(end + 1)  # right-open cap
    # Now produce [lo, hi] inclusive intervals
    win_lo = edges[:-1]
    win_hi = [x - 1 for x in edges[1:]]
    # Trim/extend to match nrows
    if len(win_lo) != nrows:
        # Best-effort: if counts don't match, re-make edges by row count
        win_lo = [start + i * winsize for i in range(nrows)]
        win_hi = [min(lo + winsize - 1, end) for lo in win_lo]
    dfw = pd.DataFrame({
        "chrom": chrom,
        "win_lo": win_lo,
        "win_hi": win_hi
    })
    dfw["L_bp"] = (dfw["win_hi"] - dfw["win_lo"] + 1).clip(lower=1).astype(int)
    return dfw

def read_windows(csv_path: str,
                 chrom: str,
                 region_start: int,
                 region_end: int,
                 winsize: int) -> pd.DataFrame:
    """
    Accepts either:
      (A) 'full' CSV with thetaW/thetaW_per_bp, rho_region, win_lo, win_hi, chrom
      (B) 'minimal' CSV with columns: SNP, thetaW, rho  (no coordinates)
    Returns a DataFrame with columns: chrom, win_lo, win_hi, L_bp, theta_region, rho_region
    """
    df = pd.read_csv(csv_path)
    cols = set(df.columns.str.lower())

    # harmonize theta column (regional)
    theta_col = None
    if "thetaw" in cols:
        theta_col = "thetaW"
    elif "thetaw_per_bp" in cols:
        theta_col = "thetaW_per_bp"
    else:
        raise ValueError("CSV must contain 'thetaW' or 'thetaW_per_bp'.")

    # harmonize rho column (regional)
    rho_col = None
    if "rho_region" in cols:
        rho_col = "rho_region"
    elif "rho" in cols:
        rho_col = "rho"
    else:
        raise ValueError("CSV must contain 'rho_region' or 'rho'.")

    have_coords = {"win_lo", "win_hi"}.issubset(cols)
    have_chrom = ("chrom" in cols)

    if have_coords and have_chrom:
        # Full CSV path
        out = df.copy()
        out["chrom"] = out["chrom"].astype(str)
        out["win_lo"] = out["win_lo"].astype(int)
        out["win_hi"] = out["win_hi"].astype(int)
        out["L_bp"] = (out["win_hi"] - out["win_lo"] + 1).clip(lower=1).astype(int)
    else:
        # Minimal CSV: construct coordinates from arguments and row count
        coords = build_windows(str(chrom), int(region_start), int(region_end),
                               int(winsize), len(df))
        out = coords

    # theta_region
    if theta_col == "thetaW_per_bp":
        out["theta_region"] = (df["thetaW_per_bp"].astype(float) * out["L_bp"].astype(int)).astype(float)
    else:
        out["theta_region"] = df["thetaW"].astype(float)

    # rho_region
    out["rho_region"] = df[rho_col].astype(float)

    # sanity checks
    if (out["theta_region"] < 0).any():
        raise ValueError("Negative theta_region encountered.")
    if (out["rho_region"] < 0).any():
        raise ValueError("Negative rho_region encountered.")
    if (out["L_bp"] <= 0).any():
        raise ValueError("Non-positive L_bp encountered.")

    return out[["chrom","win_lo","win_hi","L_bp","theta_region","rho_region"]]

def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout

def hard_sweep_args():
    tau   = float(np.random.uniform(0.01, 0.2))
    alpha = float(np.random.uniform(200, 2000))
    x     = float(np.random.uniform(0.1, 0.9))
    return ["-ws", f"{tau:.4f}", "-a", f"{alpha:.2f}", "-x", f"{x:.3f}"]

def bneck_args():
    t_drop    = float(np.random.uniform(0.05, 0.3))
    size_drop = float(np.random.uniform(0.05, 0.5))
    t_rec     = t_drop + float(np.random.uniform(0.05, 0.4))
    size_rec  = float(np.random.uniform(0.7, 1.2))
    return ["-en", f"{t_drop:.4f}", "0", f"{size_drop:.3f}",
            "-en", f"{t_rec:.4f}", "0", f"{size_rec:.3f}"]

def base_name(row) -> str:
    return f"chr{row['chrom']}_{int(row['win_lo'])}_{int(row['win_hi'])}"

def main():
    p = argparse.ArgumentParser(description="Generate ms files per window (SWEEP vs NOSWEEP) using discoal.")
    p.add_argument("--csv", required=True, help="Windows CSV (thetaW/thetaW_per_bp, rho or rho_region).")
    p.add_argument("--idx", type=int, required=True, help="0-based row index to process.")
    p.add_argument("--discoal", required=True, help="Path to discoal binary.")
    p.add_argument("--out_dir", default="ms_out", help="Directory to write .ms files.")
    p.add_argument("--n_chrom", type=int, default=50, help="Number of haplotypes (nsam).")
    p.add_argument("--reps_sweep", type=int, default=10, help="Replicates per window for SWEEP.")
    p.add_argument("--reps_other", type=int, default=10, help="Replicates per window for NOSWEEP.")
    p.add_argument("--seed", type=int, default=2025, help="Base seed.")

    # For minimal CSVs (no coords), provide region info:
    p.add_argument("--chrom", default="21", help="Chromosome label to embed in filenames (default: 21).")
    p.add_argument("--region-start", type=int, default=10326675, help="Region start (1-based inclusive).")
    p.add_argument("--region-end",   type=int, default=46709983, help="Region end (1-based inclusive).")
    p.add_argument("--winsize",      type=int, default=100000, help="Window size (bp) if constructing coordinates.")

    args = p.parse_args()

    # preflight
    if os.path.isdir(args.discoal):
        sys.exit(f"Path is a directory, not a binary: {args.discoal}")
    if not (os.path.isfile(args.discoal) and os.access(args.discoal, os.X_OK)):
        sys.exit(f"discoal not executable: {args.discoal}")

    # deterministic per-task randomness
    rng_seed = args.seed + args.idx
    random.seed(rng_seed); np.random.seed(rng_seed)
    seed1 = (rng_seed & 0x7fffffff) or 1
    seed2 = ((rng_seed * 1103515245 + 12345) & 0x7fffffff) or 2
    dseed = ["-d", str(seed1), str(seed2)]

    # read CSV + (if needed) construct window coords
    df = read_windows(args.csv, args.chrom, args.region_start, args.region_end, args.winsize)

    if not (0 <= args.idx < len(df)):
        raise IndexError(f"idx {args.idx} out of range 0..{len(df)-1}")

    row   = df.iloc[args.idx]
    theta = float(row["theta_region"])
    rho   = float(row["rho_region"])
    nsite = int(row["L_bp"])
    base  = base_name(row)
    ensure_dir(args.out_dir)

    # --- SWEEP (hard) ---
    cmd_sweep = [args.discoal, str(args.n_chrom), str(args.reps_sweep), str(nsite),
                 "-t", f"{theta:.6g}", "-r", f"{rho:.6g}", *hard_sweep_args(), *dseed]
    txt_sweep = run(cmd_sweep)
    with open(os.path.join(args.out_dir, f"{base}.SWEEP.ms"), "w") as f:
        f.write(txt_sweep)

    # --- NOSWEEP (neutral OR bottleneck) ---
    if random.random() < 0.5:
        extra2 = []                 # neutral
        label2 = "NOSWEEP_neutral"
    else:
        extra2 = bneck_args()       # demographic confounder
        label2 = "NOSWEEP_bneck"

    cmd_other = [args.discoal, str(args.n_chrom), str(args.reps_other), str(nsite),
                 "-t", f"{theta:.6g}", "-r", f"{rho:.6g}", *extra2, *dseed]
    txt_other = run(cmd_other)
    with open(os.path.join(args.out_dir, f"{base}.{label2}.ms"), "w") as f:
        f.write(txt_other)

    print(f"[OK] {base}: theta={theta:.3f} rho={rho:.3f} nsites={nsite} nsam={args.n_chrom} -> SWEEP + {label2}")

if __name__ == "__main__":
    main()
