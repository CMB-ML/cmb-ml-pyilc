import numpy as np


def gaussian_beam_profile(fwhm_arcmin: float, ell_max: int) -> np.ndarray:
    """
    Return Gaussian beam B_ell for the given FWHM (arcmin).
    Used for harmonic-space rebeaming.
    """
    fwhm_rad = np.deg2rad(fwhm_arcmin / 60.0)
    sigma = fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    ell = np.arange(ell_max + 1, dtype=float)
    return np.exp(-0.5 * ell * (ell + 1.0) * sigma**2)


def compute_fwhm_pix_per_scale(ell_per_scale: np.ndarray) -> np.ndarray:
    """
    Compute the per-scale pixel-space FWHM smoothing (deg) from needlet ells.
    Matches PyILC's FWHM_pix calculation.
    """
    fwhm_pix = (180.0 / np.pi) * np.sqrt(8.0 * np.log(2.0)) / np.maximum(ell_per_scale, 1.0)
    if len(fwhm_pix) > 0:
        fwhm_pix[0] *= 2.0  # broader smoothing for lowest scale
    return fwhm_pix

def compute_fwhm_pix_per_scale_pyilc_style(
    N_modes: np.ndarray,
    N_freqs_to_use: np.ndarray,
    ILC_bias_tol: float,
    N_deproj: int = 0,
) -> np.ndarray:
    """
    Compute FWHM_pix[i] per scale (in degrees) following PyILC's bias-tolerance criterion.
    """
    # sigma_pix = np.sqrt(
    #     np.abs(2.0 * ((N_deproj + 1) - N_freqs_to_use) / (N_modes * ILC_bias_tol))
    # )
    # sigma_pix = np.clip(sigma_pix, 1e-10, np.pi)  # sanity clamp

    sigma_pix_temp = np.sqrt(np.abs(2.0 * ((N_deproj + 1) - N_freqs_to_use)
                                    / (N_modes * ILC_bias_tol)))
    # FWHM_pix[i] = np.sqrt(8.*np.log(2.)) * sigma_pix_temp  # radians

    fwhm_pix_rad = np.sqrt(8.0 * np.log(2.0)) * sigma_pix_temp
    fwhm_pix_deg = np.degrees(fwhm_pix_rad)
    return fwhm_pix_deg