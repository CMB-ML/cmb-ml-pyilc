import os
from typing import Optional
import healpy as hp
import numpy as np
from .config import Config
from .executors.A_waveletize import WaveletizeExecutor
from .executors.B_covariance import CovarianceExecutor
from .executors.C_weights import WeightsExecutor
from .executors.D_prediction import PredictionExecutor
from cnilc_pipeline.utils.gaussian_utils import compute_fwhm_pix_per_scale, compute_fwhm_pix_per_scale_pyilc_style


def load_maps(paths, remove_dipole=False):
    maps = []
    for p in paths:
        m = hp.read_map(p, field=0, dtype=np.float64)
        m = m * 1e-6  # Very lazy conversion to K_CMB
        if remove_dipole:
            m = hp.remove_dipole(m)
        maps.append(m)
    return maps


def load_mask(path, nside_ref=None):
    if path is None:
        return None
    m = hp.read_map(path, field=0, dtype=np.int64)
    if nside_ref is not None and hp.get_nside(m) != nside_ref:
        m = hp.ud_grade(m, nside_out=nside_ref)
    return m


def run_pipeline(cfg: Config, out_dir: str = "outputs") -> np.ndarray:
    os.makedirs(out_dir, exist_ok=True)

    maps = load_maps(cfg.freq_map_files, cfg.remove_dipole)
    base_nside = cfg.N_side
    mask = load_mask(cfg.mask_before_covariance_computation, nside_ref=base_nside)

    wave = WaveletizeExecutor(cfg)
    coeffs_per_scale = wave.run(maps)  # list[scale] -> list[coeff_map for active freqs]
    ell_per_scale = wave.needlets.ell
    # fwhm_pix_per_scale = compute_fwhm_pix_per_scale(ell_per_scale)

    # Save needlet coefficient maps (PyILC style) using ORIGINAL channel indices
    active_per_scale = getattr(wave.needlets, 'active_freqs_per_scale', [])
    for s, coeffs_at_scale in enumerate(coeffs_per_scale):
        if active_per_scale:
            orig_idxs = active_per_scale[s]
        else:
            orig_idxs = list(range(len(coeffs_at_scale)))
        for f_local, coeff in enumerate(coeffs_at_scale):
            f_orig = orig_idxs[f_local]
            fname = os.path.join(out_dir, f"CN__needletcoeffmap_freq{f_orig}_scale{s}.fits")
            hp.write_map(fname, coeff, overwrite=True, dtype=np.float64)

    freqs_to_use = wave.needlets.freqs_to_use

    cov_exec = CovarianceExecutor(cfg, mask)
    w_exec = WeightsExecutor(cfg)

    Sigmas = []
    weights = []

    for s, coeffs_at_scale in enumerate(coeffs_per_scale):
        # ell_min_per_scale = [np.min(np.where(f > 0)) for f in wave.needlets.filters]
        # ell_max_per_scale = [np.max(np.where(f > 0)) for f in wave.needlets.filters]
        # N_modes = (np.array(ell_max_per_scale) + 1)**2 - np.array(ell_min_per_scale)**2
        N_modes = np.sum((2.*wave.needlets.ell + np.ones(wave.needlets._wv.ELLMAX+1)) * (wave.needlets.filters[s])**2.)

        N_deproj = 0  # CMB-only. TODO: Put this somewhere more ... blergh
        fwhm_pix_deg = compute_fwhm_pix_per_scale_pyilc_style(N_modes=N_modes, 
                                                              N_freqs_to_use=len(coeffs_at_scale),
                                                              ILC_bias_tol=cfg.ILC_bias_tol,
                                                              N_deproj=N_deproj)
        # Ensure mask matches coeff resolution
        coeff_nside = hp.get_nside(coeffs_at_scale[0])
        # local_mask = None
        # if mask is not None and hp.get_nside(mask) != coeff_nside:
        #     local_mask = hp.ud_grade(mask, nside_out=coeff_nside)
        # else:
        #     local_mask = mask

        # Compute global Sigma and per-pair product maps
        # prod_maps = cov_exec.run(coeffs_at_scale, freqs_to_use, this_scale=s, fwhm_deg=fwhm_pix_per_scale[s])
        prod_maps = cov_exec.run(
                        coeffs_at_scale,
                        freqs_to_use,
                        this_scale=s,
                        fwhm_deg=fwhm_pix_deg,
                        orig_idxs=active_per_scale[s],
                    )
        # Sigmas.append(Sigma)

        # Save each element of covariance as FITS (per freq pair) with ORIGINAL channel ids
        if active_per_scale:
            orig_idxs = active_per_scale[s]
        else:
            orig_idxs = list(range(len(coeffs_at_scale)))
        for (i_orig, j_orig), prod_map in prod_maps.items():
            fname = os.path.join(
                out_dir,
                f"CN__needletcoeff_covmap_freq{i_orig}_freq{j_orig}_scale{s}.fits"
            )
            hp.write_map(fname, prod_map, overwrite=True, dtype=np.float64)

    #     # Solve weights using ONLY the active freqs for this scale
    #     w = w_exec.run(Sigma, active_freq_indices=orig_idxs)
    #     weights.append(w)

    #     includechannels = ''.join(str(i) for i in orig_idxs)
    #     fname = os.path.join(out_dir, f"CN_needletILCmap_scale{s}_component_{cfg.ILC_preserved_comp}_includechannels{includechannels}.fits")
    #     # Construct ILC map at this scale from active coeffs
    #     X = np.vstack(coeffs_at_scale).T  # (n_pix, n_active)
    #     y = X @ w  # (n_pix,)
    #     hp.write_map(fname, y, overwrite=True, dtype=np.float64)

    # # Final cleaned map (all scales combined)
    # pred = PredictionExecutor(cfg, wave.needlets)
    # cleaned_map = pred.run(coeffs_per_scale, weights)

    # final_name = os.path.join(out_dir, f"CN_needletILCmap_component_{cfg.ILC_preserved_comp}.fits")
    # hp.write_map(final_name, cleaned_map, overwrite=True, dtype=np.float64)
    # hp.write_map(cfg.output_path, cleaned_map, overwrite=True, dtype=np.float64)

    # return cleaned_map


# def run_pipeline(cfg: Config, out_dir: str = "outputs") -> np.ndarray:
#     os.makedirs(out_dir, exist_ok=True)

#     # ---- Load maps (with dipole removal and units fix) ----
#     maps = load_maps(cfg.freq_map_files, cfg)

#     # ---- Load optional masks ----
#     mask_wavelet = None
#     if getattr(cfg, "mask_before_wavelet_computation", None) is not None:
#         mask_wavelet = hp.read_map(cfg.mask_before_wavelet_computation, field=0, dtype=np.float64)

#     mask_cov = None
#     if getattr(cfg, "mask_before_covariance_computation", None) is not None:
#         mask_cov = hp.read_map(cfg.mask_before_covariance_computation, field=0, dtype=np.float64)

#     # ---- Waveletize ----
#     wave = WaveletizeExecutor(cfg)
#     if mask_wavelet is not None:
#         # downgrade/upgrade mask to input Nside if needed
#         if hp.get_nside(mask_wavelet) != cfg.N_side:
#             mask_wavelet = hp.ud_grade(mask_wavelet, nside_out=cfg.N_side)
#         # apply before waveletizing
#         maps = [m * mask_wavelet for m in maps]
#     coeffs_per_scale = wave.run(maps)

#     # ---- Covariance ----
#     cov_exec = CovarianceExecutor(cfg)
#     Sigma = []
#     for s, coeffs_at_scale in enumerate(coeffs_per_scale):
#         local_mask = None
#         if mask_cov is not None:
#             # match resolution of this scale
#             if hp.get_nside(mask_cov) != hp.get_nside(coeffs_at_scale[0]):
#                 local_mask = hp.ud_grade(mask_cov, nside_out=hp.get_nside(coeffs_at_scale[0]))
#             else:
#                 local_mask = mask_cov
#         Sigma.append(cov_exec.run(coeffs_at_scale, mask=local_mask))

#     # ---- Weights ----
#     w_exec = WeightsExecutor(cfg)

#     active_per_scale = getattr(wave.needlets, 'active_freqs_per_scale', [])

#     Sigmas = []
#     weights = []
#     for s, coeffs_at_scale in enumerate(coeffs_per_scale):
#         # Ensure mask matches coeff resolution
#         coeff_nside = hp.get_nside(coeffs_at_scale[0])
#         local_mask = None
#         if mask_cov is not None and hp.get_nside(mask_cov) != coeff_nside:
#             local_mask = hp.ud_grade(mask_cov, nside_out=coeff_nside)
#         else:
#             local_mask = mask_cov

#         # Compute global Sigma and per-pair product maps
#         Sigma, prod_maps = cov_exec.run(coeffs_at_scale, mask=local_mask)
#         Sigmas.append(Sigma)

#         # Save each element of covariance as FITS (per freq pair) with ORIGINAL channel ids
#         if active_per_scale:
#             orig_idxs = active_per_scale[s]
#         else:
#             orig_idxs = list(range(len(coeffs_at_scale)))
#         for (i_local, j_local), prod_map in prod_maps.items():
#             i_orig, j_orig = orig_idxs[i_local], orig_idxs[j_local]
#             fname = os.path.join(out_dir, f"CN__needletcoeff_covmap_freq{i_orig}_freq{j_orig}_scale{s}.fits")
#             hp.write_map(fname, prod_map, overwrite=True, dtype=np.float64)

#         # Solve weights using ONLY the active freqs for this scale
#         w = w_exec.run(Sigma, active_freq_indices=orig_idxs)
#         weights.append(w)

#         includechannels = ''.join(str(i) for i in orig_idxs)
#         fname = os.path.join(out_dir, f"CN_needletILCmap_scale{s}_component_{cfg.ILC_preserved_comp}_includechannels{includechannels}.fits")
#         # Construct ILC map at this scale from active coeffs
#         X = np.vstack(coeffs_at_scale).T  # (n_pix, n_active)
#         y = X @ w  # (n_pix,)
#         hp.write_map(fname, y, overwrite=True, dtype=np.float64)

#     # ---- Prediction ----
#     pred = PredictionExecutor(cfg, wave.needlets)
#     cleaned_map = pred.run(coeffs_per_scale, weights)

#     # ---- Save final map ----
#     final_name = os.path.join(out_dir, f"CN_needletILCmap_component_{cfg.ILC_preserved_comp}.fits")
#     hp.write_map(final_name, cleaned_map, overwrite=True, dtype=np.float64)

#     return cleaned_map


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise RuntimeError("Usage: python -m simpler_ilc.run <config.yaml>")
    cfg = Config.from_yaml(sys.argv[1])
    cleaned = run_pipeline(cfg)
    print("Pipeline finished. Cleaned map ready:", cleaned.shape)