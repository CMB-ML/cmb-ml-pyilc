from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class Config:
    # Geometry / transforms
    N_side: int
    ELLMAX: int
    ellmin: int
    ellpeaks: List[int]

    # Frequency setup
    N_freqs: int
    freqs_delta_ghz: List[Optional[float]]  # use None for non-CMB channels
    beam_FWHM_arcmin: List[float]
    freq_map_files: List[str]

    # ILC
    ILC_preserved_comp: str = "CMB"
    ILC_deproj_comps: List[str] = field(default_factory=list)
    ILC_bias_tol: float = 1e-3

    # Resolution to perform ILC
    perform_ILC_at_beam: Optional[float] = None  # arcmin

    # Masks & I/O
    mask_before_covariance_computation: Optional[str] = None  # path to FITS (or None)
    output_path: str = "./cnilc_out.fits"

    work_in_car: bool = False
    work_in_healpix: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
