# src/stabilizer_entropy.py
from __future__ import annotations
import numpy as np

# Restrict the stabilizer generators to the specified region
def restrict_stabilizers(stabs: np.ndarray, region: list[int], n_qubits: int) -> np.ndarray:
    A = np.array(sorted(set(region)), dtype=int)
    cols = list(A) + list(A + n_qubits)
    return stabs[:, cols].copy()

# Compute the reduced row echelon form over GF(2), which is modulo 2
def gf2_rref(M: np.ndarray):
    M = M.copy() % 2
    m, n = M.shape
    row = 0
    pivots = 0
    # Iterate over columns to find pivots, columns 
    for col in range(n):
        pivot = None
        for r in range(row, m):
            # Find a row with a leading 1 in the current column
            if M[r, col] == 1:
                # Found a pivot
                pivot = r
                break
        if pivot is None:
            continue
        # Swap the current row with the pivot row
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        # Eliminate all other 1s in this column
        for r in range(m):
            if r != row and M[r, col] == 1:
                M[r, :] ^= M[row, :]
        row += 1
        pivots += 1
        # Stop if we've processed all rows
        if row == m:
            break
    # return the RREF matrix and the number of pivots found
    return M, pivots

# Construct the symplectic form matrix for n qubits
def symplectic_form(n: int) -> np.ndarray:
    # Create a 2n x 2n symplectic form matrix
    I = np.eye(n, dtype=np.uint8)
    # Construct the block matrix
    Z = np.zeros_like(I)
    # Top half: [0 I]
    # Bottom half: [I 0]
    top = np.concatenate([Z, I], axis=1)
    bot = np.concatenate([I, Z], axis=1)
    # Combine top and bottom halves
    return np.concatenate([top, bot], axis=0).astype(np.uint8)

# Calculate the number of ebits across the cut defined by the restricted stabilizers
def ebits_across_cut(restricted_stabs: np.ndarray) -> int:
    # Restrict stabilizers to region A, compute symplectic form, and find rank
    SA = restricted_stabs % 2
    m, w = SA.shape
    nA = w // 2
    # If there are no qubits in region A, return 0 ebits
    if nA == 0:
        return 0
    J = symplectic_form(nA)
    # Compute K = SA * J * SA^T over GF(2)
    K = (SA @ J) % 2
    K = (K @ SA.T) % 2
    # Compute the rank of K over GF(2)
    _, rK = gf2_rref(K)
    # Calculate the number of ebits
    e = (2*nA - rK) // 2
    return int(e)

# Compute the entanglement entropy of a specified region given the stabilizers
def entropy_of_region(stabs: np.ndarray, region: list[int], n_qubits: int) -> int:
    SA = restrict_stabilizers(stabs, region, n_qubits)
    return ebits_across_cut(SA)