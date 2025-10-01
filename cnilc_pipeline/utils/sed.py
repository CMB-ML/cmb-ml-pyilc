from typing import List, Optional
import numpy as np

def get_sed_vector(freqs_delta_ghz: List[Optional[float]], comp: str) -> np.ndarray:
    comp = comp.upper()
    if comp in ("CMB", "KSZ"):
        a = np.ones(len(freqs_delta_ghz), dtype=float)
        for i, f in enumerate(freqs_delta_ghz):
            if f is None:
                a[i] = 0.0
        return a
    raise NotImplementedError(f"SED '{comp}' not supported in minimal pipeline")
