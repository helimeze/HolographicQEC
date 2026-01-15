# HolographicQEC

The layout:
- General **stabilizer entropy** engine (imported from the basic repo).
- **Correlation-augmented decoding** (add edges/weights from detector MI).
- A small **holographic entanglement** notebook: AdS3 geodesic length for an interval
  (Ryu–Takayanagi) with UV-cutoff, for comparison to area-law/constant terms in codes.

## Workflow
### 1) Augmented decoding
```bash
python src/build_circuit.py --rounds 80 --p 1.5e-3 --out circuit.stim
python src/sample_and_dem.py --in circuit.stim --shots 200000 --shots_out shots.npy --dem_out model.dem
python src/correlation_analysis.py --shots shots.npy --pairs 15000 --out mi.parquet
python src/decode_augmented.py --dem model.dem --shots shots.npy --mi mi.parquet --mi_threshold 0.02
```

### 2) Holographic EE (AdS3 geodesic)
Run the simple Python notebook script:
```bash
python notebooks/holographic_RT_AdS3.py
```
It computes the (UV) regulated geodesic lenghth for a boundary interval of size l at cutoff epsilon,
showing S = (c/3) * log(l/epsilon) behavior, and compares qualitative constant terms to TEE. Here c is the central charge as per uusual..


# Pipeline

Minimal, runnable pipeline for **(A)** topological entanglement entropy (TEE) of a CSS (stabilizer) code state
and **(B)** syndrome correlation analysis + decoding with **Stim** and **PyMatching**.

## Features
- Compute von Neumann entanglement entropy for stabilizer states via **symplectic Gaussian elimination**.
- Estimate **topological entanglement entropy** (Kitaev–Preskill) on a toy toric-like CSS state.
- Build a small Stim circuit (multi-round toy surface-like fragment) with noise, sample detection events.
- Compute **pairwise detector mutual information**, export Parquet/CSV.
- Decode with **PyMatching** 

## Install
```bash
python -m venv .venv && source .venv/bin/activate  
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quickstart (TEE)
```bash
python src/tee_demo.py --L 5 --layout toric_like --kp_square 1,1,2 2,2,2 1,3,2
```
This prints TEE estimate \(\gamma\) in bits.

## Quickstart (Syndrome correlations + decoding)
```bash
# Build circuit
python src/build_circuit.py --rounds 50 --p 2e-3 --out circuit.stim

# Sample shots (to .npy) and export DEM
python src/sample_and_dem.py --in circuit.stim --shots 100000 --shots_out shots.npy --dem_out model.dem

# Correlation analysis
python src/correlation_analysis.py --shots shots.npy --pairs 5000 --out mi.parquet

# Decode
python src/decode_basic.py --dem model.dem --shots shots.npy
```

## Notes
- The TEE demo uses a **toy** toric-like stabilizer set for clarity.
- For production surface codes, plug in your stabilizer generators or Clifford circuits and reuse `stabilizer_entropy.py`.
