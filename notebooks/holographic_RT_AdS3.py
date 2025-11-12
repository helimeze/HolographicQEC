# notebooks/holographic_RT_AdS3.py
import numpy as np, matplotlib.pyplot as plt

def S_interval(ell, eps, c=1.0):
    ell = np.asarray(ell, dtype=float)
    return (c/3.0) * np.log(ell/eps)

ell = np.linspace(0.1, 10.0, 200)
eps_list = [0.05, 0.1, 0.2]
plt.figure()
for eps in eps_list:
    plt.plot(ell, S_interval(ell, eps, c=1.0), label=f"epsilon={eps}")
plt.xlabel("interval length l"); plt.ylabel("S(l) (units of c)")
plt.title("AdS3 RT: S = (c/3) log(l/epsilon)")
plt.legend()
plt.tight_layout()
plt.savefig("rt_ads3_interval.png", dpi=160)
print("Saved rt_ads3_interval.png")
