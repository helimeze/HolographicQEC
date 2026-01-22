# src/decode_basic.py
import argparse, numpy as np
import pymatching as pm

# Basic decoding of measurement shots using a detector error model (DEM)
if __name__ == "__main__":
    
    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    
    # Required DEM file and shots file
    ap.add_argument("--dem", type=str, required=True)
    ap.add_argument("--shots", type=str, required=True)
    # Optional: If no logical predictions are available, use a proxy
    args = ap.parse_args()
    # Load the matching decoder from the DEM file
    m = pm.Matching.from_detector_error_model_file(args.dem)
    # Load the measurement
    shots = np.load(args.shots)
    # Decode the shots to get logical predictions
    pred = m.decode_batch(shots)
    # If no predictions are available, compute a proxy logical error rate
    if pred.size == 0:
        proxy = shots.sum(axis=1) % 2
        pL = proxy.mean()
        print(f"(Proxy) logical-like parity error rate: {pL:.6f}")
    # Otherwise, compute and print the logical error rate
    else:
        logical_flip = pred.any(axis=1)
        print(f"Logical error rate: {logical_flip.mean():.6f}")