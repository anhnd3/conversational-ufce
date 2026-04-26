#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
from typing import List, Dict

# Import the author table from our reproduction script
import sys
ROOT = os.path.abspath(os.getcwd())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.reproduce_results_v3 import AUTHOR_TABLE7, ALL_DATASETS, METRICS, METHODS

def process_run_type(run_type: str, run_dir: str, root_out: str):
    print(f"Processing run type: {run_type} in {run_dir}")
    
    long_rows = []
    
    for dataset in ALL_DATASETS:
        # Find the summary CSV
        pattern = os.path.join(run_dir, f"summary_{dataset}_*.csv")
        matches = glob.glob(pattern)
        if not matches:
            print(f"  [WARN] No summary CSV found for {dataset} at {pattern}")
            continue
        
        csv_path = matches[0]
        try:
            df = pd.read_csv(csv_path, index_col=0) # Index is 'method' or 'metric'
        except Exception as e:
            print(f"  [ERROR] Could not read {csv_path}: {e}")
            continue
            
        # The CSV has metrics as rows and methods as columns
        for method in METHODS:
            if method not in df.columns:
                continue
            
            for metric in METRICS:
                if metric not in df.index:
                    continue
                    
                ours = df.loc[metric, method]
                author = float("nan")
                
                # Fetch author data if available
                ds_author = AUTHOR_TABLE7.get(dataset)
                if ds_author and method in ds_author and metric in ds_author[method]:
                    author = ds_author[method][metric]
                
                delta = ours - author if pd.notna(author) and pd.notna(ours) else float("nan")
                
                long_rows.append({
                    "dataset": dataset,
                    "run_type": run_type,
                    "method": method,
                    "metric": metric,
                    "ours": ours,
                    "author": author,
                    "delta": delta
                })
                
    if not long_rows:
        print(f"  [WARN] No valid data found for run type {run_type}")
        return

    # Create and save the long dataframe
    long_df = pd.DataFrame(long_rows)
    out_file = os.path.join(root_out, f"final_table7_{run_type}_long.csv")
    long_df.to_csv(out_file, index=False)
    print(f"Saved consolidated table to: {out_file}")
    
    # Also create a delta-only table (filtered for non-NaN authors)
    delta_df = long_df.dropna(subset=["author"]).copy()
    if not delta_df.empty:
        delta_file = os.path.join(root_out, f"final_table7_delta_vs_author_{run_type}.csv")
        delta_df.to_csv(delta_file, index=False)
        print(f"Saved delta comparison to: {delta_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the timestamped run directory (e.g., outputs/final_table7_freeze/20260425_165918)")
    args = parser.parse_args()
    
    root_out = args.run_dir
    if not os.path.isdir(root_out):
        print(f"[ERROR] Run directory not found: {root_out}")
        sys.exit(1)
        
    raw_dir = os.path.join(root_out, "raw")
    if os.path.isdir(raw_dir):
        process_run_type("raw", raw_dir, root_out)
    else:
        print(f"[WARN] Raw directory not found: {raw_dir}")
        
    strict_dir = os.path.join(root_out, "strict_validity")
    if os.path.isdir(strict_dir):
        process_run_type("strict_validity", strict_dir, root_out)
    else:
        print(f"[WARN] Strict directory not found: {strict_dir}")

if __name__ == "__main__":
    main()
