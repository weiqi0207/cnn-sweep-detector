#!/usr/bin/env python3
import argparse, gzip, re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def read_vcf_table(vcf_path: str) -> tuple[pd.DataFrame, list[str]]:
    # Parse header for column names
    with gzip.open(vcf_path, "rt") as fh:
        header = None
        for line in fh:
            if line.startswith("#CHROM"):
                header = line.strip().split("\t")
                break
    if header is None:
        raise SystemExit("VCF missing #CHROM header.")
    # Load fixed + sample columns (entire file)
    df = pd.read_csv(
        vcf_path, sep="\t", comment="#", header=None, names=header,
        compression="gzip", dtype=str, low_memory=False
    )
    # Keep biallelic SNPs (single-base REF/ALT, no commas)
    is_snp = (
        df["REF"].str.len().eq(1) &
        df["ALT"].str.len().eq(1) &
        df["REF"].str.match("^[ACGT]$", na=False) &
        df["ALT"].str.match("^[ACGT]$", na=False)
    )
    df = df.loc[is_snp, :].copy()
    df["POS"] = df["POS"].astype(int)
    sample_cols = [c for c in df.columns if c not in
                   ["#CHROM","CHROM","POS","ID","REF","ALT","QUAL","FILTER","INFO","FORMAT"]]
    return df, sample_cols

def window_edges(pos_min: int, pos_max: int, winsize: int) -> np.ndarray:
    start = ((pos_min - 1) // winsize) * winsize + 1
    end   = pos_max + 1
    edges = np.arange(start, end + winsize, winsize, dtype=int)
    if edges[-1] < end: edges = np.append(edges, end)
    return edges

def df_to_haplotype_matrix(dfw: pd.DataFrame, sample_cols: list[str]) -> np.ndarray:
    """Return (2*n_samples, S) uint8 with 0=REF, 1=ALT per haplotype per site."""
    S = len(dfw)
    if S == 0: return np.zeros((0,0), dtype=np.uint8)
    n_samples = len(sample_cols)
    H = np.zeros((2 * n_samples, S), dtype=np.uint8)
    # iterate sites (columns of matrix)
    for j, (_, row) in enumerate(dfw.iterrows()):
        for i, s in enumerate(sample_cols):
            gt = str(row[s]).split(":", 1)[0]
            if gt in (".", "./.", ".|."):
                a, b = "0", "0"
            else:
                alleles = re.split(r"[\/|]", gt)
                if len(alleles) < 2:
                    a = alleles[0] if alleles and alleles[0] != "." else "0"
                    b = "0"
                else:
                    a = alleles[0] if alleles[0] != "." else "0"
                    b = alleles[1] if alleles[1] != "." else "0"
            H[2*i,   j] = 1 if a != "0" else 0
            H[2*i+1, j] = 1 if b != "0" else 0
    return H

def sort_haplotypes_by_similarity(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    if n <= 2: return M
    order = [0]; used = np.zeros(n, bool); used[0] = True; cur = 0
    for _ in range(1, n):
        cand = np.where(~used)[0]
        d = np.array([np.count_nonzero(M[cur] ^ M[j]) for j in cand])
        jmin = cand[int(np.argmin(d))]
        order.append(jmin); used[jmin] = True; cur = jmin
    return M[np.array(order), :]

def fit_width_columns(M: np.ndarray, width: int) -> np.ndarray:
    n, S = M.shape
    if S == width: return M
    if S == 0:    return np.zeros((max(1,n), width), dtype=np.uint8)
    if S > width:
        idx = np.linspace(0, S-1, num=width).round().astype(int)
        return M[:, idx]
    out = np.zeros((n, width), dtype=np.uint8); out[:, :S] = M; return out

def to_png_array(M: np.ndarray, invert: bool=False) -> np.ndarray:
    A = M if invert else (1 - M)   # default: 0→white, 1→black
    return (A * 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser(description="Convert VCF to fixed-width haplotype PNGs (one per 100kb window).")
    ap.add_argument("--vcf", required=True, help="Path to .vcf.gz")
    ap.add_argument("--outdir", required=True, help="Output folder for PNGs")
    ap.add_argument("--winsize", type=int, default=100_000, help="Window size (bp)")
    ap.add_argument("--width", type=int, default=512, help="Target columns (sites) per image")
    ap.add_argument("--sort", action="store_true", help="Sort haplotypes by similarity")
    ap.add_argument("--invert", action="store_true", help="Invert colors (0=black,1=white)")
    args = ap.parse_args()

    outdir = Path(args.outdir); ensure_dir(outdir)
    df, sample_cols = read_vcf_table(args.vcf)
    if df.empty:
        raise SystemExit("No biallelic SNPs found after filtering.")

    chrom = str(df["#CHROM"].iloc[0])
    edges = window_edges(int(df["POS"].min()), int(df["POS"].max()), args.winsize)

    n_written = 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        dfw = df[(df["POS"] >= lo) & (df["POS"] < hi)]
        # Skip windows with 0 SNPs
        if dfw.empty:
            continue
        M = df_to_haplotype_matrix(dfw, sample_cols)  # (2N, S)
        if M.size == 0:  # paranoid
            continue
        if args.sort:
            M = sort_haplotypes_by_similarity(M)
        fixed = fit_width_columns(M, args.width)
        arr = to_png_array(fixed, invert=args.invert)
        img = Image.fromarray(arr, mode="L")
        outpath = outdir / f"chr{chrom}_{lo}_{hi}.png"
        img.save(outpath, compress_level=1)
        n_written += 1
        if (n_written % 50) == 0:
            print(f"[{n_written}] up to {outpath.name}")

    print(f"[OK] Wrote {n_written} images to {outdir}")

if __name__ == "__main__":
    main()