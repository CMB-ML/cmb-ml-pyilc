"""
This script runs a pipeline for prediction and analysis of the cleaned CMB signal using <TODO: Name it>

The pipeline consists of the following steps:
1. <TODO: Steps here>

And also generating various analysis figures, throughout.

Final comparison is performed in the main_analysis_compare.py script.

Usage:
    python main_<TODO: Name it>.py
"""
import logging

import hydra

from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.sims import MaskCreatorExecutor
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
# from pyilc_local.B_predict_executor import UsePyILCExecutor
# from pyilc_local.C_make_ps import PyILCMakePSExecutor
from cmbml.sims import (
    MaskCreatorExecutor, 
    HalfMissionNoiseExecutor,
    SimHMCreatorExecutor
    )
from cmbml.planck_as_sim import (
    ObsMapsConvertExecutor, 
    ObsHMMapsConvertExecutor
    )
from pyilc_local.B_predict_executor import (
    GetILCWeightsExecutor,
    ApplyWeightsExecutor
    )
from pyilc_local.G_fg_only_maps import (
    MakeDebiasFGCfgExecutor, 
    MakeDebiasFGMapExecutor
    )
from pyilc_local.J_ps_with_debias import DebiasPredPowerSpectrumExecutor


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="cfg", config_name="config_pyilc")
def run_pyilc(cfg):
    logger.debug(f"Running train in {__file__}")
    pipes = [
            # HydraConfigCheckerExecutor,
            MaskCreatorExecutor,
            HalfMissionNoiseExecutor,
            SimHMCreatorExecutor,
            MakeDebiasFGCfgExecutor,
            MakeDebiasFGMapExecutor,
            GetILCWeightsExecutor,
            ApplyWeightsExecutor,
            DebiasPredPowerSpectrumExecutor,
            ]
    run(cfg, pipes)


@hydra.main(version_base=None, config_path="cfg", config_name="config_pyilc_on_planck")
def run_on_planck(cfg):
    logger.debug(f"Running train in {__file__}")
    pipes = [
            # ObsMapsConvertExecutor,
            # ObsHMMapsConvertExecutor,
            GetILCWeightsExecutor,
            ApplyWeightsExecutor,
            DebiasPredPowerSpectrumExecutor,
            ]
    run(cfg, pipes)


def run(cfg, pipes):
    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)
    for pipe in pipes:
        pipeline_context.add_pipe(pipe)

    pipeline_context.prerun_pipeline()
    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        logger.exception("An exception occured during the pipeline.", exc_info=e)
        raise e
    finally:
        logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()


if __name__ == "__main__":
    # run_pyilc()
    run_on_planck()
