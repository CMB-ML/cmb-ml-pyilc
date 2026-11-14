"""
This script runs a pipeline for cleaning CMB signal using PyILC.

The pipeline consists of the following steps:
1. Creating masks
2. Generating half-mission noise and observations
3. Generating foreground debiasing realizations
4. Computing CNILC weights and predictions
5. Applying weights to half-mission and foreground maps
6. Estimating debiased power spectra

Because settings in PyILC cause conflicts for Matplotlib, no figure-producing executors should be imported here or errors will result.

The script uses the Hydra library for configuration management.

Usage:
    python main_pyilc_predict.py
"""
import logging
import hydra
from omegaconf import OmegaConf


OmegaConf.register_new_resolver(
    "append", lambda items, item: list(items) + [item]
)


from cmbml.core import (
    PipelineContext,
    LogMaker
    )
from cmbml.sims.ex import (
    MaskCreatorExecutor, 
    HalfMissionNoiseExecutor,
    SimHMCreatorExecutor
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
def main(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(MaskCreatorExecutor)
    pipeline_context.add_pipe(HalfMissionNoiseExecutor)
    pipeline_context.add_pipe(SimHMCreatorExecutor)
    pipeline_context.add_pipe(MakeDebiasFGCfgExecutor)
    pipeline_context.add_pipe(MakeDebiasFGMapExecutor)
    pipeline_context.add_pipe(GetILCWeightsExecutor)
    pipeline_context.add_pipe(ApplyWeightsExecutor)
    pipeline_context.add_pipe(DebiasPredPowerSpectrumExecutor)

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
    main()
