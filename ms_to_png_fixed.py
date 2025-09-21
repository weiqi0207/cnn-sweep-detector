#!/usr/bin/env python3
import argparse, os, re
from pathlib import Path
import numpy as np
from PIL import Image

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def parse_ms(inpath: Path):
    """
    Yield (rep_id, positions01, hapxsite) per replicate.
    hapxsite: (n_hap, S) uint8 in {0,1}
    positions01: (S,) float in [0,1) from 'positions:' line (if present), else None
    """
    with open(inpath, "r") as fh:
        rep = -1
        hap_lines = []
        nsites = None
        positions = None
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("//"):
                if hap_lines:
                    mat = np.array([[int(c) for c in row] for row in hap_lines], dtype=np.uint8)
                    yield rep, positions, mat
                    hap_lines, positions, nsites = [], None, None
                rep += 1
                continue
            if line.startswith("segsites:"):
                nsites = int(line.split(":")[1].strip())
                continue
            if line.startswith("positions:"):
                pos = line.split(":")[1].strip().split()
                positions = np.array([float(p) for p in pos], dtype=np.float32)
                continue
            # haplotype rows (strings of 0/1 length nsites)
            if (nsites is not None) and re.fullmatch(r"[01]{%d}" % nsites, line):
                hap_lines.append(line)
        if hap_lines:
            mat = np.array([[int(c) for c in row] for row in hap_lines], dtype=np.uint8)
            yield rep, positions, mat

def sort_haplotypes_by_similarity(M: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbor ordering (rows)."""
    n = M.shape[0]
    if n <= 2: return M
    order = [0]
    used = np.zeros(n, dtype=bool); used[0] = True
    cur = 0
    for _ in range(1, n):
        cand = np.where(~used)[0]
        # Hamming distance via XOR on uint8
        d = np.array([np.count_nonzero(M[cur] ^ M[j]) for j in cand])
        jmin = cand[int(np.argmin(d))]
        order.append(jmin); used[jmin] = True; cur = jmin
    return M[np.array(order), :]

def fit_width_columns(M: np.ndarray, positions01, width: int) -> np.ndarray:
    """
    Return (n_hap, width) by resampling columns uniformly.
    If S > width: choose indices via linspace over sites (not random).
    If S < width: right-pad with zeros.
    """
    n, S = M.shape
    if S == width:
        return M
    if S == 0:
        return np.zeros((max(1, n), width), dtype=np.uint8)
    if S > width:
        # Index-based uniform sampling is fine; using positions gives same result if positions are ~uniform.
        idx = np.linspace(0, S - 1, num=width).round().astype(int)
        return M[:, idx]
    # S < width: right-pad
    out = np.zeros((n, width), dtype=np.uint8)
    out[:, :S] = M
    return out

def to_png_array(M: np.ndarray, invert: bool=False) -> np.ndarray:
    """
    Map 0/1 to grayscale. Default: 0→white (255), 1→black (0).
    Set invert=True for 0→black, 1→white.
    """
    A = M if invert else (1 - M)
    return (A * 255).astype(np.uint8)

def guess_label_from_name(path: Path) -> str:
    name = path.name.upper()
    if "SWEEP" in name and "NOSWEEP" not in name:
        return "SWEEP"
    if "NOSWEEP" in name:
        return "NOSWEEP"
    return "UNKNOWN"

def guess_window_from_name(path: Path):
    # Expect names like chr21_12345_22345.*.ms → return ("21", 12345, 22345)
    m = re.search(r"chr(\w+)_([0-9]+)_([0-9]+)", path.name, flags=re.IGNORECASE)
    if not m:
        return ("NA", None, None)
    chrom = m.group(1)
    lo = int(m.group(2)); hi = int(m.group(3))
    return (chrom, lo, hi)

def main():
    ap = argparse.ArgumentParser(description="Convert ms-style files to fixed-width haplotype PNGs.")
    ap.add_argument("--ms", required=True, help="Path to an ms-style output (.ms)")
    ap.add_argument("--outdir", required=True, help="Output folder for PNGs")
    ap.add_argument("--width", type=int, default=1200, help="Target number of site columns per image")
    ap.add_argument("--sort", action="store_true", help="Sort haplotypes (rows) by similarity")
    ap.add_argument("--invert", action="store_true", help="Invert colors (0→black,1→white)")
    args = ap.parse_args()

    inpath = Path(args.ms)
    outdir = Path(args.outdir); ensure_dir(outdir)
    label = guess_label_from_name(inpath)
    chrom, win_lo, win_hi = guess_window_from_name(inpath)

    n_written = 0
    for rep, pos01, M in parse_ms(inpath):
        # Handle empty segsites
        if M.size == 0 or M.shape[1] == 0:
            fixed = np.zeros((max(1, M.shape[0]), args.width), dtype=np.uint8)
        else:
            if args.sort:
                M = sort_haplotypes_by_similarity(M)
            fixed = fit_width_columns(M, pos01, args.width)

        arr = to_png_array(fixed, invert=args.invert)
        img = Image.fromarray(arr, mode="L")

        # Compose informative filename
        if chrom is not None and win_lo is not None and win_hi is not None:
            stem = f"chr{chrom}_{win_lo}_{win_hi}_rep{rep:04d}_{label}.png"
        else:
            stem = f"{inpath.stem}_rep{rep:04d}_{label}.png"

        img.save(outdir / stem, compress_level=1)
        n_written += 1

    print(f"[OK] Wrote {n_written} image(s) to {outdir}")

if __name__ == "__main__":
    main()