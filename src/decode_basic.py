# src/decode_basic.py
import argparse, numpy as np
import pymatching as pm

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", type=str, required=True)
    ap.add_argument("--shots", type=str, required=True)
    args = ap.parse_args()
    m = pm.Matching.from_detector_error_model_file(args.dem)
    shots = np.load(args.shots)
    pred = m.decode_batch(shots)
    if pred.size == 0:
        proxy = shots.sum(axis=1) % 2
        pL = proxy.mean()
        print(f"(Proxy) logical-like parity error rate: {pL:.6f}")
    else:
        logical_flip = pred.any(axis=1)
        print(f"Logical error rate: {logical_flip.mean():.6f}")