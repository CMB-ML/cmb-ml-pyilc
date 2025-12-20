import numpy as np
try:
    from pyilc.wavelets import jit_linsolve as _jit_linsolve
    from pyilc.wavelets import jit_linsolve_parallelb as _jit_linsolve_parallelb
    from pyilc.wavelets import jit_det as _jit_det
    from pyilc.wavelets import jit_matmul as _jit_matmul
except Exception:
    _jit_linsolve = _jit_linsolve_parallelb = _jit_det = _jit_matmul = None

def batched_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if _jit_matmul is None:
        return np.einsum("...ik,...kj->...ij", A, B)
    return _jit_matmul(A, B)

def batched_det(A: np.ndarray) -> np.ndarray:
    if _jit_det is None:
        return np.array([np.linalg.det(a) for a in A])
    return _jit_det(A)

def solve_weights_numba(Sigma: np.ndarray, a: np.ndarray) -> np.ndarray:
    if _jit_linsolve is None and _jit_linsolve_parallelb is None:
        print("Using backup solve_weights_numba()")
        inv_Sa = np.linalg.solve(Sigma, a)
        alpha = float(a.T @ inv_Sa)
        return (inv_Sa / alpha).reshape(-1)
    Sig_b = Sigma[None, :, :]
    a_b = a[None, :]
    if _jit_linsolve_parallelb is not None:
        inv_Sa_b = _jit_linsolve_parallelb(Sig_b, a_b)
    else:
        inv_Sa_b = _jit_linsolve(Sig_b, a_b)
    alpha_b = np.einsum("bi,bi->b", a_b, inv_Sa_b)
    w_b = inv_Sa_b / alpha_b[:, None]
    return w_b[0]

def compute_covariance(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    n = max(1, Xc.shape[0] - 1)
    return (Xc.T @ Xc) / n
