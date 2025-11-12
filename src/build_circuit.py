# src/build_circuit.py
import argparse, stim

def toy_surface_fragment(rounds=20, p=1e-3):
    c = stim.Circuit()
    n = 9
    data = [stim.QubitTarget(i) for i in range(n)]
    c.append_operation("DEPOLARIZE1", data, p)
    for t in range(rounds):
        anc = stim.GateTarget( n + t )
        for q in [0,1,3,4]:
            c.append_operation("CZ", [stim.QubitTarget(q), anc])
        c.append_operation("MZ", [anc])
        c.append_operation("DETECTOR", [stim.GateTarget.rec(-1)])
        c.append_operation("X_ERROR", [stim.GateTarget.rec(-1)], p)
    return c

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--p", type=float, default=2e-3)
    ap.add_argument("--out", type=str, default="circuit.stim")
    args = ap.parse_args()
    circ = toy_surface_fragment(rounds=args.rounds, p=args.p)
    open(args.out,"w").write(str(circ))
    print(f"Wrote {args.out}")
