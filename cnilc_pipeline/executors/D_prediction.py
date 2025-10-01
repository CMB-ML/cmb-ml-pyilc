from typing import List
import numpy as np

class PredictionExecutor:
    def __init__(self, cfg, needlets_adapter):
        self.cfg = cfg
        self.needlets = needlets_adapter

    def run(self, coeffs_per_scale: List[List[np.ndarray]], weights_per_scale: List[np.ndarray]) -> np.ndarray:
        cleaned_coeffs = []
        for scale_idx, coeffs_at_scale in enumerate(coeffs_per_scale):
            X = np.vstack(coeffs_at_scale).T  # (n_pix, n_freq)
            w = weights_per_scale[scale_idx]
            y = X @ w  # (n_pix,)
            cleaned_coeffs.append(y)
        return self.needlets.inverse(cleaned_coeffs, nside=self.cfg.N_side)
