# simpler_ilc/utils/needlets.py
from typing import List, Tuple
import numpy as np
import healpy as hp
# Import directly from installed pyilc package
from pyilc.wavelets import Wavelets, waveletize, synthesize

class SurrogateInfo:
    """Minimal surrogate to satisfy pyilc.waveletize asserts."""
    def __init__(self, cfg):
        self.work_in_car = cfg.work_in_car
        self.work_in_healpix = cfg.work_in_healpix
        self.N_side = cfg.N_side
        self.ELLMAX = cfg.ELLMAX
        self.ellmin = cfg.ellmin
        self.ellpeaks = cfg.ellpeaks
        self.N_freqs = cfg.N_freqs
        self.freqs_delta_ghz = cfg.freqs_delta_ghz
        self.beam_FWHM_arcmin = cfg.beam_FWHM_arcmin
        self.ILC_preserved_comp = cfg.ILC_preserved_comp
        self.ILC_deproj_comps = cfg.ILC_deproj_comps
        self.ILC_bias_tol = cfg.ILC_bias_tol
        # PyILC knobs with sane defaults if absent in cfg
        self.wavelet_beam_criterion = getattr(cfg, "wavelet_beam_criterion", 1e-3)
        self.drop_channels = getattr(cfg, "drop_channels", []) or []
        self.override_N_freqs_to_use = getattr(cfg, "override_N_freqs_to_use", False)
        self.N_freqs_to_use = getattr(cfg, "N_freqs_to_use", None)  # list[int] length = N_scales, if override True
        self.N_deproj = getattr(cfg, "N_deproj", 0)
        self.N_side_to_use = None
        self.print_timing = True

# ---- Helpers that mirror PyILC scale selection logic ----

def _gaussian_beam_profile(fwhm_arcmin: float, ell_max: int) -> np.ndarray:
    """Return Gaussian beam B_ell for the given FWHM (arcmin)."""
    fwhm_rad = (fwhm_arcmin / 60.0) * (np.pi / 180.0)
    sigma = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    ell = np.arange(ell_max + 1, dtype=float)
    return np.exp(-0.5 * ell * (ell + 1.0) * sigma * sigma)

def _compute_ell_F_per_scale(wv: Wavelets, wavelet_beam_criterion: float) -> np.ndarray:
    """Per-scale ℓ_F where the filter crosses the criterion on its decreasing side (as in PyILC)."""
    ell_F = np.zeros(wv.N_scales, dtype=int)
    for i in range(wv.N_scales):
        filt = wv.filters[i]
        ell_peak = int(np.argmax(filt))
        sub = np.abs(filt[ell_peak:] - wavelet_beam_criterion)
        ell_F[i] = min(ell_peak + int(np.argmin(sub)), wv.ELLMAX)
    if wv.N_scales > 1:
        # PyILC uses the penultimate criterion for the last scale
        ell_F[-1] = ell_F[-2]
    return ell_F

def compute_nsides_per_scale_from_ellF(base_nside: int, ell_F: np.ndarray) -> List[int]:
    """Smallest power-of-two N_side strictly larger than ℓ_F[i], capped at base_nside."""
    nsides: List[int] = []
    for val in ell_F:
        ns = 2
        for j in range(2, 20):            # up to N_side = 2^19
            if val < 2 ** j:
                ns = 2 ** j
                break
        if ns > base_nside:
            ns = base_nside
        nsides.append(int(ns))
    return nsides

def compute_freqs_to_use(wv: Wavelets, info: SurrogateInfo) -> np.ndarray:
    """
    Return boolean [n_scales x n_freqs] of (scale, freq) pairs to use,
    following PyILC: keep when ℓ_F[i] <= ℓ_B[j], excluding drop_channels.
    If override_N_freqs_to_use is set, keep only the highest-resolution channels requested.
    """
    n_scales, n_freqs = wv.N_scales, info.N_freqs
    freqs_to_use = np.full((n_scales, n_freqs), False)

    ell_F = _compute_ell_F_per_scale(wv, info.wavelet_beam_criterion)

    # Per-frequency ℓ_B from Gaussian beams
    beams_Bell = [_gaussian_beam_profile(info.beam_FWHM_arcmin[j], wv.ELLMAX) for j in range(n_freqs)]
    ell_B = np.array([int(np.argmin(np.abs(B - info.wavelet_beam_criterion))) for B in beams_Bell], dtype=int)

    for i in range(n_scales):
        for j in range(n_freqs):
            if ell_F[i] <= ell_B[j] and (j not in info.drop_channels):
                freqs_to_use[i, j] = True

    # Optional override: keep only the last k channels (assumed highest-res)
    if info.override_N_freqs_to_use and info.N_freqs_to_use is not None:
        for i in range(n_scales):
            k = int(info.N_freqs_to_use[i])
            keep = np.zeros(n_freqs, dtype=bool)
            if k > 0:
                keep[-k:] = True
            freqs_to_use[i, :] = freqs_to_use[i, :] & keep

    return freqs_to_use

class NeedletAdapter:
    def __init__(self, ellmin: int, ellpeaks: List[int], ellmax: int, cfg=None):
        self._wv = Wavelets(N_scales=len(ellpeaks)+1, ELLMAX=ellmax, tol=1e-6, taper_width=0)
        self.ellmin = ellmin
        self.ellpeaks = ellpeaks
        self.ellmax = ellmax
        self.ell, self.filters = self._wv.CosineNeedlets(ellmin=ellmin, ellpeaks=np.asarray(ellpeaks))
        self._cfg = cfg
        self._surrogate = SurrogateInfo(cfg) if cfg is not None else None
        self.nside_out = cfg.N_side if cfg is not None else None
        self.active_freqs_per_scale: List[List[int]] = []  # record which freqs survive at each scale

    def forward(self, maps: List[np.ndarray], nside: int):
        # Compute per-scale N_side via ℓ_F, and (scale,freq) selection matrix via ℓ_B vs ℓ_F.
        ell_F = _compute_ell_F_per_scale(self._wv, self._surrogate.wavelet_beam_criterion)
        nsides = compute_nsides_per_scale_from_ellF(nside, ell_F)
        if self._surrogate is not None:
            self._surrogate.N_side_to_use = nsides
        freqs_to_use = compute_freqs_to_use(self._wv, self._surrogate)

        # Allocate output structure: list over scales; each has the coeff arrays for the freqs that passed.
        per_scale: List[List[np.ndarray]] = [[] for _ in range(self._wv.N_scales)]
        self.active_freqs_per_scale = [[] for _ in range(self._wv.N_scales)]

        # Waveletize each frequency map; select only the pairs that PyILC would keep
        all_coeffs_per_freq: List[List[np.ndarray]] = []
        for m in maps:
            coeffs = waveletize(inp_map=m, wv=self._wv, info=self._surrogate, N_side_to_use=nsides)
            all_coeffs_per_freq.append(coeffs)

        for f, coeffs in enumerate(all_coeffs_per_freq):
            for s in range(len(coeffs)):
                if freqs_to_use[s, f]:
                    per_scale[s].append(coeffs[s])
                    self.active_freqs_per_scale[s].append(f)

        return per_scale

    def inverse(self, coeffs_per_scale: List[np.ndarray], nside: int) -> np.ndarray:
        # Use the same per-scale N_side logic
        ell_F = _compute_ell_F_per_scale(self._wv, self._surrogate.wavelet_beam_criterion)
        nsides = compute_nsides_per_scale_from_ellF(nside, ell_F)
        if self._surrogate is not None:
            self._surrogate.N_side_to_use = nsides
        return synthesize(coeffs_per_scale, wv=self._wv, N_side_out=self.nside_out)
