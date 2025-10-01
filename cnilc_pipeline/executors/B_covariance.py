from typing import List, Optional, Dict, Tuple
import numpy as np
import healpy as hp
from ..utils.numba_kernels import compute_covariance

class CovarianceExecutor:
    def __init__(self, cfg):
        self.cfg = cfg

    def run(self, coeffs_at_scale: List[np.ndarray], mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, Dict[Tuple[int,int], np.ndarray]]:
        """
        Compute (1) the *global* covariance matrix Sigma across active freqs for this scale,
        and (2) the *pixelwise* covariance/product maps for each (i,j) pair.

        Returns: (Sigma, prod_maps)
          - Sigma: (n_active, n_active) covariance matrix
          - prod_maps: dict[(i,j)] -> map (n_pix,) with i<=j, using the *local* coeff resolution
        """
        n_active = len(coeffs_at_scale)
        if n_active == 0:
            return np.zeros((0,0), dtype=np.float64), {}

        # Pixel selection
        if mask is None:
            sel = slice(None)
        else:
            sel = (mask > 0)

        # Global covariance (for weight solve)
        if mask is None:
            X = np.vstack(coeffs_at_scale).T  # (n_pix, n_active)
        else:
            X = np.vstack([c[sel] for c in coeffs_at_scale]).T
        Sigma = compute_covariance(X).astype(np.float64, copy=False)

        # Per-pair product maps (store i<=j)
        prod_maps: Dict[Tuple[int,int], np.ndarray] = {}
        for i in range(n_active):
            ci = coeffs_at_scale[i]
            for j in range(i, n_active):
                cj = coeffs_at_scale[j]
                prod = (ci * cj).astype(np.float64, copy=False)
                if mask is not None:
                    prod = prod * (mask > 0)
                prod_maps[(i, j)] = prod
        return Sigma, prod_maps
