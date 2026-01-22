# src/tee_demo.py
import argparse, numpy as np
from stabilizer_entropy import entropy_of_region

# define the toric-like code stabilizers
def toric_like_stabilizers(L: int):
    def edge_id(i, j, ori):
        i %= L; j %= L
        return (i*L + j) + (0 if ori==0 else L*L)
    n = 2 * L * L
    xs, zs = [], []
    for i in range(L):
        for j in range(L):
            qubits = [edge_id(i, j, 0), edge_id(i, j-1, 0),
                      edge_id(i, j, 1), edge_id(i-1, j, 1)]
            x = np.zeros(n, dtype=np.uint8); x[qubits] = 1
            z = np.zeros(n, dtype=np.uint8)
            xs.append(x); zs.append(z)
    for i in range(L):
        for j in range(L):
            qubits = [edge_id(i, j, 0), edge_id(i, j+1, 1),
                      edge_id(i+1, j, 0), edge_id(i, j, 1)]
            x = np.zeros(n, dtype=np.uint8)
            z = np.zeros(n, dtype=np.uint8); z[qubits] = 1
            xs.append(x); zs.append(z)
    X = np.stack(xs); Z=np.stack(zs)
    stabs = np.concatenate([X, Z], axis=1)
    return stabs, n

# parse the KP regions from command line argument
def parse_kp_regions(arg: str, L: int):
    parts = arg.strip().split()
    regions = []
    for p in parts:
        i0,j0,side = map(int, p.split(','))
        qs=set()
        for i in range(i0, i0+side):
            for j in range(j0, j0+side):
                qs.add((i%L)*L + (j%L))
                qs.add(L*L + (i%L)*L + (j%L))
        regions.append(sorted(qs))
    if len(regions)!=3:
        raise ValueError("Provide exactly three regions for KP TEE.")
    return regions

# main execution: compute TEE for toric-like code
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--layout", type=str, default="toric_like", choices=["toric_like"])
    ap.add_argument("--kp_square", type=str, default="1,1,2 2,2,2 1,3,2")
    args = ap.parse_args()
    
    # generate stabilizers
    stabs, n = toric_like_stabilizers(args.L)
    A,B,C = parse_kp_regions(args.kp_square, args.L)
    def S(R): return entropy_of_region(stabs, R, n)

    # compute entropies and TEE
    SA, SB, SC = S(A), S(B), S(C)
    SAB = S(sorted(set(A)|set(B)))
    SBC = S(sorted(set(B)|set(C)))
    SCA = S(sorted(set(C)|set(A)))
    SABC = S(sorted(set(A)|set(B)|set(C)))
    # compute TEE using Kitaev–Preskill formula
    gamma = SA + SB + SC - SAB - SBC - SCA + SABC
    print(f"S(A)={SA}, S(B)={SB}, S(C)={SC}")
    print(f"S(AB)={SAB}, S(BC)={SBC}, S(CA)={SCA}, S(ABC)={SABC}")
    print(f"TEE (Kitaev–Preskill) gamma ≈ {gamma} bits")