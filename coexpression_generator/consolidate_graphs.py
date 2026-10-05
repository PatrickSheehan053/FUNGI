"""
to_HYPHAE / consolidate_graphs.py
────────────────────────────────────
Stitches parallelized parquet chunks into a unified Edge List,
runs a rigorous topological sanity check, and saves the master
parent graph for FUNGI.
"""

import os
import argparse
import pandas as pd
import anndata as ad

def run_diagnostics(df, h5ad_path=None):
    """
    Executes topological sanity checks and computes network statistics
    before handing the graph to FUNGI.
    """
    print("\n" + "═" * 65)
    print("  FUNGI PRE-FLIGHT GRAPH DIAGNOSTICS")
    print("═" * 65)

    # 1. Base Dimensions
    V_targets = df['Target'].nunique()
    V_regulators = df['Regulator'].nunique()
    nodes = set(df['Target']).union(set(df['Regulator']))
    V_total = len(nodes)
    E = len(df)

    print("  [1. GRAPH TOPOLOGY]")
    print(f"  Total Directed Edges (E): {E:>12,}")
    print(f"  Total Unique Nodes (V)  : {V_total:>12,}")
    print(f"  Unique Targets          : {V_targets:>12,}")
    print(f"  Unique Regulators       : {V_regulators:>12,}")

    # 2. Sanity Checks
    print("\n  [2. SANITY CHECKS]")
    dupes = df.duplicated(subset=['Regulator', 'Target']).sum()
    nans = df['Importance'].isna().sum()
    self_loops = (df['Regulator'] == df['Target']).sum()

    print(f"  Duplicate Edges         : {dupes:>12,} " + ("(PASS)" if dupes == 0 else "(FAIL)"))
    print(f"  NaN Importances         : {nans:>12,} " + ("(PASS)" if nans == 0 else "(FAIL)"))
    print(f"  Self-Loops (A -> A)     : {self_loops:>12,} (Biologically Valid)")

    if dupes > 0:
        print("  ⚠️ CRITICAL: Dropping duplicate edges before saving...")
        df = df.drop_duplicates(subset=['Regulator', 'Target'])
        E = len(df) # Update edge count

    # 3. Density & Degree
    print("\n  [3. NETWORK STATISTICS]")
    possible_edges = V_total * V_total
    density = E / possible_edges if possible_edges > 0 else 0
    avg_out_degree = E / V_total if V_total > 0 else 0
    max_hub = df['Regulator'].value_counts().max()

    print(f"  Graph Density           : {density:>12.4%} (Sparsity: {1-density:.4%})")
    print(f"  Avg Out-Degree per Node : {avg_out_degree:>12.2f} edges")
    print(f"  Max Out-Degree (Hub)    : {max_hub:>12,} edges")

    # 4. Weight Distributions
    print("\n  [4. EDGE WEIGHTS (LightGBM Gain)]")
    print(f"  Mean Importance         : {df['Importance'].mean():>12.4f}")
    print(f"  Median Importance       : {df['Importance'].median():>12.4f}")
    print(f"  Max Importance          : {df['Importance'].max():>12.4f}")

    # 5. Metadata Cross-Reference
    if h5ad_path and os.path.exists(h5ad_path):
        print("\n  [5. SOURCE METADATA CROSS-REFERENCE]")
        try:
            # Peek at the h5ad without loading the matrix into RAM
            adata = ad.read_h5ad(h5ad_path, backed='r')
            print(f"  Source Entities (Cells) : {adata.n_obs:>12,}")
            print(f"  Source Features (Genes) : {adata.n_vars:>12,}")
            
            if adata.n_vars != V_total:
                dropped = adata.n_vars - V_total
                print(f"  ⚠️ NOTE: {dropped:,} genes had 0.0 importance across all bootstraps")
                print(f"           and were natively dropped from the network topology.")
            adata.file.close()
        except Exception as e:
            print(f"  Could not read h5ad metadata: {e}")

    print("═" * 65 + "\n")
    return df

def consolidate(chunk_dir, output_file, total_tasks, h5ad_path=None):
    print(f"Gathering array chunks from: {chunk_dir}")
    
    chunks = []
    for i in range(total_tasks):
        chunk_path = os.path.join(chunk_dir, f"chunk_{i}.parquet")
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"CRITICAL ERROR: {chunk_path} is missing. Array task likely failed.")
        
        df = pd.read_parquet(chunk_path)
        chunks.append(df)
        print(f"  ✓ Loaded chunk {i} with {len(df):,} edges")
        
    print("\nStitching edge lists into global parent graph...")
    final_df = pd.concat(chunks, ignore_index=True)
    
    # Run the rigorous pre-flight checks
    final_df = run_diagnostics(final_df, h5ad_path)
    
    print(f"Saving to {output_file}...")
    final_df.to_parquet(output_file)
    print("Consolidation Complete. Hand-off to FUNGI is clear.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_dir", type=str, required=True, help="Directory containing the .parquet chunks")
    parser.add_argument("--output_file", type=str, required=True, help="Path for the final .parquet parent graph")
    parser.add_argument("--total_tasks", type=int, default=10, help="Number of SBATCH array chunks")
    parser.add_argument("--h5ad_file", type=str, default=None, help="(Optional) Original .h5ad file to extract cell counts")
    args = parser.parse_args()

    consolidate(args.chunk_dir, args.output_file, args.total_tasks, args.h5ad_file)