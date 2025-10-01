from typing import List
import numpy as np
from ..config import Config
from ..utils.sed import get_sed_vector
from ..utils.numba_kernels import solve_weights_numba

class WeightsExecutor:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, Sigma: np.ndarray, active_freq_indices: List[int]) -> np.ndarray:
        """
        Solve ILC weights for the *active channels at this scale*.
        `active_freq_indices` are the original channel indices kept at this scale.
        """
        # Build SED only for active channels
        a_full = get_sed_vector(self.cfg.freqs_delta_ghz, comp=self.cfg.ILC_preserved_comp).astype(np.float64)
        a = a_full[np.array(active_freq_indices, dtype=int)]
        # Shape checks
        if Sigma.shape[0] != Sigma.shape[1] or Sigma.shape[0] != a.shape[0]:
            raise ValueError(f"Sigma/a shape mismatch: Sigma {Sigma.shape}, a {a.shape}, active {active_freq_indices}")
        return solve_weights_numba(Sigma, a)
