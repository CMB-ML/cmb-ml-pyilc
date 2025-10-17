from typing import List, Optional, Dict, Tuple
import numpy as np
import healpy as hp
from ..utils.numba_kernels import compute_covariance


class CovarianceExecutor:
    def __init__(self, cfg, mask):
        self.cfg = cfg
        self.mask = mask
        print(self.mask.dtype)

    def run(self, coeffs_at_scale, freqs_to_use, this_scale, fwhm_deg=None, orig_idxs=None):
        cov_maps_temp = {}
        nside = hp.get_nside(coeffs_at_scale[0])
        fwhm_rad = np.deg2rad(fwhm_deg)
        # fwhm_rad = np.radians(fwhm_deg) / np.sqrt(8*np.log(2))
        # fwhm_rad = fwhm_deg

        print(f"Smoothing by {fwhm_rad} for {this_scale} scale")

        # Mask prep
        mask = self.mask
        hp.write_map(f"outputs/raw_mask_scale{this_scale}.fits", self.mask)
        if mask is not None:
            dgraded_mask = hp.ud_grade(mask, nside)
            dgraded_mask[dgraded_mask!=0]=1
            hp.write_map(f"outputs/degraded_mask_scale{this_scale}.fits", dgraded_mask)
            if np.sum(dgraded_mask)==0:
                raise ValueError(f"Mask, when downgraded to {nside}, masks all pixels.")
            smoothed_mask = hp.sphtfunc.smoothing(dgraded_mask, fwhm_rad)
            hp.write_map(f"outputs/smoothed_mask_scale{this_scale}.fits", smoothed_mask)
            fskyinv = np.zeros(smoothed_mask.shape)
            fskyinv[smoothed_mask!=0] = 1.0/smoothed_mask[smoothed_mask!=0]
            fskyinv[smoothed_mask==0] = 1.0e100
            hp.write_map(f"outputs/fsky_inv_scale{this_scale}.fits", fskyinv)
        else:
            dgraded_mask = fskyinv = 1.0

        # CONFIRMED TO BE IDENTICAL ABOVE HERE. MINOR DEVIATIONS (1e-10 in smoothed maps)
        # Significant deviation in covariance maps

        # Un/smoothed maps
        smoothed_maps_A, unsmoothed_maps_A = {}, {}

        for i_local, a_global in enumerate(orig_idxs):
            if freqs_to_use[this_scale][a_global]:
                mA = dgraded_mask * coeffs_at_scale[i_local]
                smoothed_maps_A[a_global] = dgraded_mask * fskyinv * hp.sphtfunc.smoothing(mA, fwhm_rad)
                hp.write_map(f"smoothed_map_scale{this_scale}_freq{a_global}.fits", smoothed_maps_A[a_global])
                unsmoothed_maps_A[a_global] = mA

        # Covariance maps
        for i_local, a_global in enumerate(orig_idxs):
            for j_local, b_global in enumerate(orig_idxs[i_local:], start=i_local):
                if freqs_to_use[this_scale][a_global] and freqs_to_use[this_scale][b_global]:
                    A, As = unsmoothed_maps_A[a_global], smoothed_maps_A[a_global]
                    B, Bs = unsmoothed_maps_A[b_global], smoothed_maps_A[b_global]
                    diffprod = (A - As) * (B - Bs)
                    cov_map = hp.sphtfunc.smoothing(diffprod, fwhm_rad) * fskyinv
                    cov_maps_temp[a_global, b_global] = cov_map

        return cov_maps_temp
