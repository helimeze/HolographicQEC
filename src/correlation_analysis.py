# src/correlation_analysis.py
import argparse, numpy as np
import pandas as pd

# Compute mutual information between two binary variables
def mutual_information_binary(x, y, eps=1e-12):
    p11 = np.mean(x & y)
    px = np.mean(x); py = np.mean(y)
    #  Calculate joint probabilities
    p10 = px - p11; p01 = py - p11; p00 = 1 - px - py + p11
    #  Calculate mutual information
    P = np.array([p00,p01,p10,p11]) + eps
    Q = np.array([(1-px)*(1-py), (1-px)*py, px*(1-py), px*py]) + eps
    return float(np.sum(P*np.log2(P/Q)))

# Main script to compute mutual information between detector pairs
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=str, required=True)
    ap.add_argument("--pairs", type=int, default=5000)
    ap.add_argument("--out", type=str, default="mi.parquet")
    args = ap.parse_args()
    shots = np.load(args.shots)

    # Generate random pairs of detectors and compute their mutual information
    S,D = shots.shape
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, D, size=(args.pairs,2))
    rows=[]
    for i,j in pairs:
        if i==j: continue
        mi = mutual_information_binary(shots[:,i], shots[:,j])
        rows.append((int(i),int(j),mi))
    
    # Save the results to a Parquet file
    df = pd.DataFrame(rows, columns=["det_i","det_j","mi"])
    df.to_parquet(args.out)
    print(f"Wrote {args.out} with {len(df)} MI pairs (D={D}, S={S})")
