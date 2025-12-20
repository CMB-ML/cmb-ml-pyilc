from typing import List, Optional
import numpy as np
from ..config import Config
# from ..utils.sed import get_sed_vector
# from ..utils.numba_kernels import solve_weights_numba
from pyilc.wavelets import jit_linsolve
from pyilc.wavelets import jit_linsolve_parallelb
from pyilc.wavelets import jit_det
from pyilc.wavelets import jit_matmul


def get_sed_vector(freqs_delta_ghz: List[Optional[float]], comp: str) -> np.ndarray:
    comp = comp.upper()
    if comp in ("CMB", "KSZ"):
        a = np.ones(len(freqs_delta_ghz), dtype=float)
        for i, f in enumerate(freqs_delta_ghz):
            if f is None:
                a[i] = 0.0
        return a
    raise NotImplementedError(f"SED '{comp}' not supported in minimal pipeline")


class WeightsExecutor:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, Sigma: np.ndarray, active_freq_indices: List[int]) -> np.ndarray:
        """
        Solve ILC weights for the *active channels at this scale*.

        Parameters
        ----------
        Sigma : np.ndarray
            Per-pixel covariance matrices for this scale.
            Shape: (n_pix, n_act, n_act), where n_act = len(active_freq_indices).
        active_freq_indices : List[int]
            Original (global) channel indices that are active at this scale.

        Returns
        -------
        w : np.ndarray
            ILC weights per pixel for the active channels at this scale.
            Shape: (n_pix, n_act).
        """
        Sigma = np.asarray(Sigma, dtype=np.float64)

        if Sigma.ndim != 3:
            raise ValueError(f"Sigma must be 3D (n_pix, n_act, n_act); got shape {Sigma.shape}")

        n_pix, n_act1, n_act2 = Sigma.shape
        if n_act1 != n_act2:
            raise ValueError(f"Sigma must be square in the last two dims; got {Sigma.shape}")

        # Full SED (one value per global frequency)
        a_full = get_sed_vector(
            self.cfg.freqs_delta_ghz,
            comp=self.cfg.ILC_preserved_comp
        ).astype(np.float64)

        # Restrict SED to the active channels at this scale
        active_freq_indices = np.asarray(active_freq_indices, dtype=int)
        a = a_full[active_freq_indices]  # shape (n_act,)

        if a.shape[0] != n_act1:
            raise ValueError(
                f"Active SED length {a.shape[0]} != Sigma n_act {n_act1}. "
                f"indices={active_freq_indices}"
            )

        # Delegate to PyILC's numba kernel.
        # Expected signature: solve_weights_numba(Sigma, a)
        #   Sigma: (n_pix, n_act, n_act)
        #   a    : (n_act,)
        # Returns:
        #   w    : (n_pix, n_act)
        w = solve_weights_numba(Sigma, a)
        return w
