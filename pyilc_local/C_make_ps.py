from omegaconf import DictConfig

from cmbml.analysis import MakePredPowerSpectrumExecutor


class PyILCMakePSExecutor(MakePredPowerSpectrumExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg, "beam_pyilc")
