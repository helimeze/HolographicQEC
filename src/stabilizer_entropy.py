# src/stabilizer_entropy.py
from __future__ import annotations
import numpy as np

def restrict_stabilizers(stabs: np.ndarray, region: list[int], n_qubits: int) -> np.ndarray:
    A = np.array(sorted(set(region)), dtype=int)
    cols = list(A) + list(A + n_qubits)
    return stabs[:, cols].copy()

def gf2_rref(M: np.ndarray):
    M = M.copy() % 2
    m, n = M.shape
    row = 0
    pivots = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if M[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        for r in range(m):
            if r != row and M[r, col] == 1:
                M[r, :] ^= M[row, :]
        row += 1
        pivots += 1
        if row == m:
            break
    return M, pivots

def symplectic_form(n: int) -> np.ndarray:
    I = np.eye(n, dtype=np.uint8)
    Z = np.zeros_like(I)
    top = np.concatenate([Z, I], axis=1)
    bot = np.concatenate([I, Z], axis=1)
    return np.concatenate([top, bot], axis=0).astype(np.uint8)

def ebits_across_cut(restricted_stabs: np.ndarray) -> int:
    SA = restricted_stabs % 2
    m, w = SA.shape
    nA = w // 2
    if nA == 0:
        return 0
    J = symplectic_form(nA)
    K = (SA @ J) % 2
    K = (K @ SA.T) % 2
    _, rK = gf2_rref(K)
    e = (2*nA - rK) // 2
    return int(e)

def entropy_of_region(stabs: np.ndarray, region: list[int], n_qubits: int) -> int:
    SA = restrict_stabilizers(stabs, region, n_qubits)
    return ebits_across_cut(SA)