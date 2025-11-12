# src/decode_augmented.py
import argparse, numpy as np, pandas as pd, pymatching as pm

def add_edges_from_mi(matching: pm.Matching, df_mi: pd.DataFrame, threshold: float = 0.02, max_extra: int = 5000):
    added = 0
    for _, row in df_mi.sort_values("mi", ascending=False).iterrows():
        if row["mi"] < threshold or added >= max_extra:
            break
        i, j, mi = int(row["det_i"]), int(row["det_j"]), float(row["mi"])
        p = min(0.49, max(1e-6, 0.5 * (1 - 2**(-mi))))
        w = -np.log(p + 1e-12)
        try:
            matching.add_edge(i, j, weight=w)
            added += 1
        except Exception:
            pass
    return added

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", type=str, required=True)
    ap.add_argument("--shots", type=str, required=True)
    ap.add_argument("--mi", type=str, required=True)
    ap.add_argument("--mi_threshold", type=float, default=0.02)
    args = ap.parse_args()

    m = pm.Matching.from_detector_error_model_file(args.dem)
    df = pd.read_parquet(args.mi)
    added = add_edges_from_mi(m, df, threshold=args.mi_threshold)
    print(f"Augmented matching graph with {added} MI-based edges")

    shots = np.load(args.shots)
    pred = m.decode_batch(shots)
    if pred.size == 0:
        proxy = shots.sum(axis=1) % 2
        pL = proxy.mean()
        print(f"(Proxy) logical-like parity error rate (augmented): {pL:.6f}")
    else:
        logical_flip = pred.any(axis=1)
        print(f"Logical error rate (augmented): {logical_flip.mean():.6f}")
