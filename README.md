# qec-entanglement-advanced

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
