# src/sample_and_dem.py
import argparse, numpy as np, stim
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--shots", type=int, default=100000)
    ap.add_argument("--shots_out", type=str, default="shots.npy")
    ap.add_argument("--dem_out", type=str, default="model.dem")
    args = ap.parse_args()
    circ = stim.Circuit.from_file(args.inp)
    sampler = circ.compile_detector_sampler()
    shots = sampler.sample(shots=args.shots)
    np.save(args.shots_out, shots)
    dem = circ.detector_error_model()
    open(args.dem_out,"w").write(str(dem))
    print(f"Saved shots -> {args.shots_out}, DEM -> {args.dem_out}, shape={shots.shape}")
