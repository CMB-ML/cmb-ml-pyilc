import logging
import shutil
from typing import List
from tqdm import tqdm

from omegaconf import DictConfig

from cmbml.core import BaseStageExecutor, Split, Asset
from cmbml.core.asset_handlers import (
    Config, 
    EmptyHandler,
    HealpyMap,
    QTableHandler 
    )
from cmbml.utils.planck_instrument import make_instrument, Instrument
from .make_pyilc_config import ILCConfigMaker
from pyilc_redir.pyilc_wrapper import run_ilc
from cmbml.utils.suppress_print import SuppressPrintErr


logger = logging.getLogger(__name__)


class UsePyILCExecutor(BaseStageExecutor):
    def __init__(self, cfg: DictConfig, stage_str="predict") -> None:
        logger.debug("Setting up to use PyILC...")
        super().__init__(cfg, stage_str=stage_str)

        self.out_config: Asset = self.assets_out.get("config_file", None)
        self.out_model: Asset = self.assets_out["model"]
        self.out_cmb_asset: Asset = self.assets_out.get("cmb_map", None)
        out_config_handler: Config
        out_model_handler: EmptyHandler
        out_cmb_map_handler: HealpyMap

        self.in_obs_assets: Asset = self.assets_in["obs_maps"]
        self.in_mask: Asset = self.assets_in.get("mask", None)
        self.in_deltabandpass: Asset = self.assets_in["deltabandpass"]
        in_obs_handler: HealpyMap
        in_deltabandpass_handler: QTableHandler

        in_det_table: Asset = self.assets_in['deltabandpass']
        # with self.name_tracker.set_context("src_root", cfg.local_system.assets_dir):
        #     planck_bandpass = self.in_deltabandpass.read()
        with self.name_tracker.set_context('src_root', cfg.local_system.assets_dir):
            det_info = in_det_table.read()

        self.instrument: Instrument = make_instrument(cfg=cfg)
        self.channels = self.instrument.dets.keys()

        self.model_cfg_maker = ILCConfigMaker(cfg, det_info)

        self.result_prefix = cfg.model.pyilc.output_prefix
        self.result_ext = cfg.model.pyilc.save_as

    def execute(self) -> None:
        self.default_execute()

    def process_split(self, 
                      split: Split) -> None:
        logger.info(f"Executing UsePyILCExecutor process_split() for split: {split.name}, for {split.n_sims} simulations.")
        for sim in tqdm(split.iter_sims()):
            with self.name_tracker.set_context("sim_num", sim):
                self.process_sim()

    def move_cmb_pred(self, destination_path):
        result_dir = self.out_model.path
        result_prefix = self.result_prefix
        result_ext = self.result_ext
        result_fn = f"{result_prefix}needletILCmap_component_CMB.{result_ext}"

        result_path = result_dir / result_fn
        result_path.rename(destination_path)

    def clear_working_directory(self):
        working_path = self.out_model.path
        for file in working_path.iterdir():
            file.unlink()
        return

    def process_sim_base(self, input_paths):
        working_path = self.out_model.path
        working_path.mkdir(exist_ok=True, parents=True)

        if self.in_mask is not None:
            mask_path = self.in_mask.path
        else:
            mask_path = None
        cfg_dict = self.model_cfg_maker.make_config(output_path=working_path,
                                                    input_paths=input_paths,
                                                    mask_path=mask_path)
        self.out_config.write(data=cfg_dict, verbose=False)
        # logger.debug("Running PyILC Code...")
        with SuppressPrintErr():
            run_ilc(self.out_config.path)

    def process_sim(self):
        if self.out_cmb_asset is None:
            raise ValueError("cmb_asset not set. This method runs when using the base class. "
                             "Either check the pipeline or your main function.")

        input_paths = []
        for freq in self.instrument.dets.keys():
            with self.name_tracker.set_context("freq", freq):
                path = self.in_obs_assets.path
                # Convert to string; we're going to convert this information to a yaml file
                input_paths.append(str(path))
        self.process_sim_base(input_paths)
        # logger.debug("Moving resulting map.")
        destination_path = self.out_cmb_asset.path
        destination_path.parent.mkdir(exist_ok=True, parents=True)

        self.move_cmb_pred(destination_path)
        self.clear_working_directory()


class GetILCWeightsExecutor(UsePyILCExecutor):
    def __init__(self, cfg, stage_str="make_weights"):
        super().__init__(cfg, stage_str)
        self.out_weight_dir_asset: Asset = self.assets_out["weight_dir"]
    
    def process_sim(self):
        input_paths = []
        for freq in self.instrument.dets.keys():
            with self.name_tracker.set_context("freq", freq):
                path = self.in_obs_assets.path
                input_paths.append(str(path))
        self.process_sim_base(input_paths)
        self.move_weights()
        self.clear_working_directory()

    def move_weights(self):
        result_dir = self.out_model.path
        result_prefix = self.result_prefix
        result_ext = self.result_ext
        weight_pattern = f"{result_prefix}weightmap_freq*_scale*_component_CMB.{result_ext}"
        for weight_path in result_dir.glob(weight_pattern):
            weight_fn = weight_path.name
            destination_path = self.out_weight_dir_asset.path / weight_fn
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            weight_path.rename(destination_path)


class ApplyWeightsExecutor(UsePyILCExecutor):
    def __init__(self, cfg: DictConfig, stage_str="apply_weights"):
        super().__init__(cfg, stage_str=stage_str)

        self.in_weight_dir: Asset = self.assets_in["weight_dir"]
        self.in_fg_maps: Asset = self.assets_in["fg_maps"]
        self.out_hm: Asset = self.assets_out["hm_wtd"]
        self.out_fg: Asset = self.assets_out["fg_wtd"]
        self.debias_split = self.splits.pop("Debias")

    def process_sim(self) -> None:
        self.process_hm(1)
        self.process_hm(2)
        for fg_num in self.debias_split.iter_sims():
            self.process_fg(fg_num)

    def process_hm(self, hm_num):
        self.copy_weights()

        obs_paths = []
        for freq in self.instrument.dets.keys():
            contexts = dict(freq=freq, hm=hm_num)
            with self.name_tracker.set_contexts(contexts):
                path = self.in_obs_assets.path
                obs_paths.append(str(path))
        self.process_sim_base(obs_paths)

        with self.name_tracker.set_context("hm", hm_num):
            self.out_hm.path.parent.mkdir(exist_ok=True, parents=True)
            self.move_cmb_pred(self.out_hm.path)
        self.clear_working_directory()

    def process_fg(self, fg_num):
        self.copy_weights()

        obs_paths = []
        for freq in self.instrument.dets.keys():
            contexts = dict(freq=freq, sim_num=fg_num)
            with self.name_tracker.set_contexts(contexts):
                path = self.in_fg_maps.path
                obs_paths.append(str(path))
        self.process_sim_base(obs_paths)

        with self.name_tracker.set_context("fg_sim", f"{fg_num:04d}"):
            self.out_fg.path.parent.mkdir(exist_ok=True, parents=True)
            self.move_cmb_pred(self.out_fg.path)
        self.clear_working_directory()

    def copy_weights(self) -> None:
        src = self.in_weight_dir.path
        if not src.is_dir():
            raise ValueError(f"Source is not a directory: {src}")

        tgt = self.out_model.path
        tgt.mkdir(exist_ok=True, parents=True)

        for f in src.iterdir():
            if f.is_file():
                # print(f"Copying \n {f} to \n {tgt/f.name}")
                shutil.copy2(f, tgt / f.name)
