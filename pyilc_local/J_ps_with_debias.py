from typing import List, Dict
import logging

from hydra.utils import instantiate
import numpy as np
from tqdm import tqdm
import healpy as hp

from omegaconf import DictConfig

from cmbml.core import (
    BaseStageExecutor, 
    Split,
    Asset
    )
# from src.analysis.make_ps import get_power as _get_power
from cmbml.core.asset_handlers.ps_handler import NumpyPowerSpectrum
from cmbml.core.asset_handlers.healpy_map_handler import HealpyMap # Import for typing hint
from cmbml.utils.physics_ps import get_auto_ps_result, get_x_ps_result, PowerSpectrum
from cmbml.utils.physics_beam import NoBeam, GaussianBeam
from cmbml.utils.physics_mask import downgrade_mask

# import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class DebiasPredPowerSpectrumExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig) -> None:
        # The following string must match the pipeline yaml
        super().__init__(cfg, stage_str="make_pred_ps_with_debias")

        self.out_auto_pred: Asset = self.assets_out.get("auto_pred", None)
        out_ps_handler: NumpyPowerSpectrum

        self.in_cmb_hm: Asset = self.assets_in["cmb_hm_map"]
        self.in_fg_map: Asset = self.assets_in["fg_map"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        self.in_mask_sm: Asset = self.assets_in.get("mask_sm", None)
        in_cmb_map_handler: HealpyMap

        # Basic parameters
        self.nside_out = self.cfg.scenario.nside
        self.lmax = cfg.model.analysis.lmax
        self.anafast_iters = cfg.model.analysis.get("ps_anafast_iters")

        # Prepare to load mask (in execute())
        self.mask_threshold = self.cfg.model.analysis.mask_threshold
        self.mask = None

        self.use_sm_mask = self.cfg.model.analysis.ps_use_smooth_mask

        # Prepare to load beam (in execute())
        # beam_type is either "beam_pyilc" or "beam_other"
        self.beam_pred = cfg.model.beam

        self.use_pixel_weights = False

        if self.cfg.map_fields != "I":
            raise NotImplementedError("Only intensity maps are currently supported.")

        self.debias_split = self.splits.pop("Debias")

    def execute(self) -> None:
        logger.debug(f"Running {self.__class__.__name__} execute().")
        self.mask = self.get_masks()
        self.beam_pred = self.get_pred_beam()
        self.default_execute()

    def get_masks(self):
        mask = None
        with self.name_tracker.set_context("src_root", self.cfg.local_system.assets_dir):
            logger.info(f"Using mask from {self.in_mask.path}")
            if self.use_sm_mask:
                mask = self.in_mask_sm.read(map_fields=self.in_mask_sm.use_fields)[0]
                if hp.npix2nside(mask.size) != self.nside_out:
                    raise ValueError("Smooth mask loaded does not match Nside of maps for analysis.")
            else:
                mask = self.in_mask.read(map_fields=self.in_mask.use_fields)[0]
                if hp.npix2nside(mask.size) != self.nside_out:
                    mask = downgrade_mask(mask, self.nside_out, threshold=self.mask_threshold)
        return mask

    def get_pred_beam(self):
        # Partially instantiate the beam object, defined in the hydra configs
        # Currently tested are GaussianBeam and NoBeam, which differ only in how they are instantiated
        beam = instantiate(self.beam_pred)
        beam = beam(lmax=self.lmax)
        return beam

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Running {self.__class__.__name__} process_split() for split: {split.name}.")
        for sim in tqdm(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def process_sim(self) -> None:
        for epoch in self.model_epochs:
            with self.name_tracker.set_context("epoch", epoch):
                self.make_pred_ps()

    def make_pred_ps(self) -> None:
        with self.name_tracker.set_context("hm", 1):
            hm1_map = self.in_cmb_hm.read()
        with self.name_tracker.set_context("hm", 2):
            hm2_map = self.in_cmb_hm.read()
        hm_x_pred_ps = get_x_ps_result(
            hm1_map,
            hm2_map,
            mask=self.mask,
            lmax=self.lmax,
            n_iter=self.anafast_iters,
            beam1=self.beam_pred,
            beam2=self.beam_pred,
            is_convolved=True)
        hm_x_pred_ps = hm_x_pred_ps.deconv_dl

        fg_ps_s = []
        for fg_num in self.debias_split.iter_sims():
            with self.name_tracker.set_context("fg_sim", f"{fg_num:04d}"):
                this_fg_map = self.in_fg_map.read()
            this_ps = get_auto_ps_result(
                this_fg_map,
                mask=self.mask,
                lmax=self.lmax,
                n_iter=self.anafast_iters,
                beam=self.beam_pred,
                is_convolved=True
            )
            fg_ps_s.append(this_ps.deconv_dl)

        fg_ps = np.mean(fg_ps_s, axis=0)

        ps = hm_x_pred_ps - fg_ps

        try:
            ps = ps.value
        except AttributeError:
            pass
        self.out_auto_pred.write(data=ps)
