"""
This script evaluates CMB temperature power spectra produced by PyILC.

The pipeline consists of the following steps:
1. Bundling theoretical and predicted power spectra
2. Selecting a polynomial smoothing degree using validation data
3. Smoothing predicted power spectra
4. Computing the mean theoretical training spectrum
5. Computing per-ell and scalar metrics
6. Estimating scalar metric uncertainties through bootstrapping

The script uses the Hydra library for configuration management.

Usage:
    python main_pyilc_ps_analysis.py
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
from cmbml.analysis.ex import (
    BundleTrueExecutor,
    BundlePredExecutor,
    GetPolyDegreeExecutor,
    BundleSmoothExecutor,
    MakeMeanSpectrumExecutor,
    GetPerEllMetricsExecutor,
    GetScalarMetricsExecutor,
    GetPerSimMetricsExecutor,
    GetScalarUncertaintyExecutor,
)


logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="cfg",
    config_name="config_pyilc_ps_analysis",
)
def main(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    # Spectrum preparation
    pipeline_context.add_pipe(BundleTrueExecutor)
    pipeline_context.add_pipe(BundlePredExecutor)
    pipeline_context.add_pipe(GetPolyDegreeExecutor)
    pipeline_context.add_pipe(BundleSmoothExecutor)

    # Spectrum evaluation
    pipeline_context.add_pipe(MakeMeanSpectrumExecutor)
    pipeline_context.add_pipe(GetPerEllMetricsExecutor)
    pipeline_context.add_pipe(GetScalarMetricsExecutor)
    pipeline_context.add_pipe(GetPerSimMetricsExecutor)
    pipeline_context.add_pipe(GetScalarUncertaintyExecutor)

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
