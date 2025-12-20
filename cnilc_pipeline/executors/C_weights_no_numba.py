# import numpy as np

# class WeightsExecutor2:
#     """
#     Compute per-pixel ILC weights for a single scale.

#     Inputs per scale:
#     - Sigma_s:   (n_pix, n_freq_s, n_freq_s)
#     - a_s:       (n_freq_s,)   mixing vector (e.g., ones for CMB)
#     - coeffs_s:  (n_freq_s, n_pix) needlet coeffs for that scale
#     """

#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.ilc_tol = 1e-12   # numeric safety

#     def run(self, Sigma_s, coeffs_s, active_freqs):
#         """
#         Sigma_s:  (n_pix, n_freq_s, n_freq_s)
#         coeffs_s: (n_freq_s, n_pix)
#         active_freqs: list of global freq indices for this scale, length n_freq_s
#         """

#         n_pix = Sigma_s.shape[0]
#         n_freq_s = len(active_freqs)

#         # CMB mixing vector a_s (PyILC uses [1,1,1,...] for temperature)
#         a_s = np.ones(n_freq_s)

#         # Output weights per pixel
#         W = np.zeros((n_freq_s, n_pix), dtype=np.float64)
#         cleaned = np.zeros(n_pix, dtype=np.float64)

#         # Loop over pixels
#         for p in range(n_pix):
#             Sigma_p = Sigma_s[p]  # (n_freq_s, n_freq_s)

#             # Inversion guard
#             try:
#                 invS = np.linalg.inv(Sigma_p)
#             except np.linalg.LinAlgError:
#                 # Regularized inverse if singular
#                 invS = np.linalg.pinv(Sigma_p)

#             # Compute numerator & denominator
#             num = invS @ a_s
#             denom = a_s @ num

#             if denom < self.ilc_tol:
#                 # fallback: uniform weights
#                 w = np.ones(n_freq_s) / n_freq_s
#             else:
#                 w = num / denom

#             # Save weights
#             W[:, p] = w

#             # Cleaned coeff (scalar at pixel p)
#             cleaned[p] = np.dot(w, coeffs_s[:, p])

#         return W, cleaned
