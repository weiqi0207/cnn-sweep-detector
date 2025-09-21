#!/usr/bin/env python3
import argparse, gzip
import numpy as np
import pandas as pd

ACCESS_CHR = "21"
ACCESS_LO  = 10_326_675
ACCESS_HI  = 46_709_983

def a1_harmonic(n: int) -> float:
    if n <= 1: return 0.0
    i = np.arange(1, n, dtype=np.float64)
    return float(np.sum(1.0 / i))

def infer_n_chrom(vcf_path: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                cols = line.strip().split("\t")
                n_samples = max(0, len(cols) - 9)  # 9 fixed cols
                return 2 * n_samples               # assume diploid
    raise SystemExit("Could not parse #CHROM header to infer sample count.")

def load_biallelic_snps(vcf_path: str) -> pd.DataFrame:
    # Read minimal columns by index: 0=#CHROM,1=POS,3=REF,4=ALT
    df = pd.read_csv(
        vcf_path, sep="\t", comment="#", header=None,
        usecols=[0,1,3,4], names=["#CHROM","POS","REF","ALT"],
        compression="gzip",
        dtype={"#CHROM":str,"POS":int,"REF":str,"ALT":str},
        low_memory=False
    )
    # Keep chr21/21 only
    df = df[df["#CHROM"].isin([ACCESS_CHR, f"chr{ACCESS_CHR}"])]
    # Restrict to accessible bounds (inclusive)
    df = df[(df["POS"] >= ACCESS_LO) & (df["POS"] <= ACCESS_HI)]
    # Keep biallelic SNPs (A/C/G/T)
    is_snp = (
        df["REF"].str.len().eq(1) &
        df["ALT"].str.len().eq(1) &
        df["REF"].str.match("^[ACGT]$", na=False) &
        df["ALT"].str.match("^[ACGT]$", na=False)
    )
    return df.loc[is_snp, ["POS"]].reset_index(drop=True)

def compute_table(vcf_path: str, winsize: int, n_chrom: int,
                  r_per_bp: float, mu_per_bp: float) -> pd.DataFrame:
    # Load SNP positions in accessible region
    dfv = load_biallelic_snps(vcf_path)
    # Build window edges on the fixed accessible bounds, half-open [lo, hi)
    start, end = ACCESS_LO, ACCESS_HI + 1
    edges = np.arange(start, end + winsize, winsize, dtype=int)
    if edges[-1] < end:
        edges = np.append(edges, end)

    # Count SNPs per window
    if dfv.empty:
        counts = np.zeros(len(edges)-1, dtype=int)
    else:
        bins = pd.cut(dfv["POS"], bins=edges, right=False, include_lowest=True)
        counts = bins.value_counts(sort=False).to_numpy(dtype=int)

    # Watterson's theta (regional) and rho (regional)
    a1 = a1_harmonic(n_chrom)
    if a1 == 0.0:
        thetaW = np.zeros_like(counts, dtype=float)
    else:
        thetaW = counts / a1
    rho = thetaW * (r_per_bp / mu_per_bp)

    # Output ONLY the requested columns
    out = pd.DataFrame({
        "SNP": counts.astype(int),
        "thetaW": thetaW.astype(float),
        "rho": rho.astype(float),
    })
    return out

def main():
    ap = argparse.ArgumentParser(description="CSV with SNPs/100kb, thetaW, rho for chr21 accessible region.")
    ap.add_argument("--vcf", required=True, help="Path to .vcf.gz")
    ap.add_argument("--winsize", type=int, default=100_000, help="Window size (bp), default 100000")
    ap.add_argument("--n-chrom", type=int, default=None, help="Override haplotypes (default: 2×#samples)")
    ap.add_argument("--r-per-bp", type=float, default=1.2e-8, help="Recombination rate per bp per gen")
    ap.add_argument("--mu-per-bp", type=float, default=1.4e-8, help="Mutation rate per bp per gen")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    n_chrom = infer_n_chrom(args.vcf, args.n_chrom)
    tbl = compute_table(args.vcf, args.winsize, n_chrom, args.r_per_bp, args.mu_per_bp)
    tbl.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(tbl)} windows over chr{ACCESS_CHR}:{ACCESS_LO}-{ACCESS_HI}.")

if __name__ == "__main__":
    main()