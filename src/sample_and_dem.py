# src/sample_and_dem.py
import argparse, numpy as np, stim

# Sample measurement shots from a Stim circuit and save the shots and detector error model to files
if __name__ == "__main__":
    # Parse command-line arguments
    ap = argparse.ArgumentParser()
    # Required input circuit file, optional output files and number of shots
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--shots", type=int, default=100000)
    ap.add_argument("--shots_out", type=str, default="shots.npy")
    ap.add_argument("--dem_out", type=str, default="model.dem")
    args = ap.parse_args()
    
    # Load the Stim circuit from file
    circ = stim.Circuit.from_file(args.inp)
    # Compile a detector sampler and sample the specified number of shots
    sampler = circ.compile_detector_sampler()
    # Generate measurement shots
    shots = sampler.sample(shots=args.shots)
    # Save the shots and detector error model to the specified output files
    np.save(args.shots_out, shots)
    # Generate and save the detector error model
    dem = circ.detector_error_model()
    # Write the DEM to file
    open(args.dem_out,"w").write(str(dem))
    print(f"Saved shots -> {args.shots_out}, DEM -> {args.dem_out}, shape={shots.shape}")
