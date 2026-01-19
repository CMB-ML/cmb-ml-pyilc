import logging
import time 
import shutil

from omegaconf import DictConfig
from tqdm import tqdm

import pysm3.units as u

from cmbml.utils.planck_instrument import make_instrument, Instrument
from cmbml.core import BaseStageExecutor, Split, Asset
from cmbml.sims import ObsCreatorExecutor, ForegroundConfigExecutor

from cmbml.core.asset_handlers.qtable_handler import QTableHandler # Import to register handler
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for VS Code hints
from cmbml.utils.fits_inspection import get_field_types_from_fits


logger = logging.getLogger(__name__)


class MakeDebiasFGCfgExecutor(ForegroundConfigExecutor):
    def __init__(self, cfg: DictConfig, stage_str="make_debias_fg_cfgs") -> None:
        super().__init__(cfg, stage_str=stage_str)


class MakeDebiasFGMapExecutor(ObsCreatorExecutor):
    def __init__(self, cfg: DictConfig, stage_str="make_debias_fg_maps") -> None:
        super().__init__(cfg, stage_str=stage_str)
