from typing import List
import numpy as np
from ..utils.needlets import NeedletAdapter

class WaveletizeExecutor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.needlets = NeedletAdapter(cfg.ellmin, cfg.ellpeaks, cfg.ELLMAX, cfg=self.cfg)

    def run(self, maps: List[np.ndarray]):
        return self.needlets.forward(maps, nside=self.cfg.N_side)
