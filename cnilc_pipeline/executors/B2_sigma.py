# cnilc_pipeline/executors/C_sigma.py
from typing import Dict, Tuple, List
import numpy as np


class SigmaExecutor:
    """
    Build the per-pixel covariance matrix Σ(p) at a given scale
    from the local covariance maps cov_maps_done.

    Σ has shape (n_pix, n_active_freqs, n_active_freqs), where
    n_active_freqs are the channels actually used at this scale.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def run(
        self,
        cov_maps_done: Dict[Tuple[int, int], np.ndarray],
        freqs_to_use_row: np.ndarray,   # freqs_to_use[this_scale], shape (N_freqs_total,)
        orig_idxs: List[int],           # global freq indices present at this scale (like in your cov executor)
    ):
        if not cov_maps_done:
            raise ValueError("SigmaExecutor.run: cov_maps_done is empty; nothing to build Σ from.")

        # Use any one map to infer n_pix
        first_map = next(iter(cov_maps_done.values()))
        n_pix = len(first_map)

        # Active *global* frequency indices at this scale
        active_global: List[int] = [f for f in orig_idxs if freqs_to_use_row[f]]
        n_act = len(active_global)
        if n_act == 0:
            raise ValueError("SigmaExecutor.run: no active frequencies at this scale.")

        # Allocate Σ(p)
        Sigma = np.zeros((n_pix, n_act, n_act), dtype=np.float64)

        # Map global freq index -> local index in [0, n_act)
        g2l = {g: i for i, g in enumerate(active_global)}

        # Fill Σ from the cov_maps_done dict
        for (i_orig, j_orig), cov_map in cov_maps_done.items():
            if i_orig not in g2l or j_orig not in g2l:
                # Pair not used at this scale (e.g. dropped by freqs_to_use)
                continue
            i_loc = g2l[i_orig]
            j_loc = g2l[j_orig]

            # cov_map is already shape (n_pix,)
            Sigma[:, i_loc, j_loc] = cov_map
            Sigma[:, j_loc, i_loc] = cov_map  # enforce symmetry

        return Sigma, active_global
