"""
This script runs a pipeline for analysis of the cleaned CMB signal generated with PyILC.

The pipeline consists of the following steps:
1. Generating per-pixel analysis results for each simulation
2. Generating per-pixel summary statistics for each simulation
3. Converting the theory power spectrum to a format that can be used for analysis
4. Generating per-ell power spectrum analysis results for each simulation
5. Generating per-ell power spectrum summary statistics for each simulation

And also generating various analysis figures, throughout.

Because settings in PyILC cause conflicts for Matplotlib, this analysis is performed separately from `main_pyilc_predict.py`.

The script uses the Hydra library for configuration management.

Usage:
    python main_pyilc_predict.py
"""
from functools import partial
import logging

import hydra

# from cmbml.utils.check_env_var import validate_environment_variable
from cmbml.core import (
                      PipelineContext,
                      LogMaker
                      )
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.sims import MaskCreatorExecutor

from cmbml.analysis import   (
                            # NILCShowSimsPostExecutor,
                            CommonRealPostExecutor,
                            CommonPredPostExecutor,
                            CommonShowSimsPostExecutor,
                            # # CommonNILCShowSimsPostIndivExecutor,
                            PixelAnalysisExecutor,
                            PixelSummaryExecutor,
                            # ConvertTheoryPowerSpectrumExecutor,
                            # MakeTheoryPSStats,
                            # # PyILCMakePSExecutor,
                            # PixelSummaryFigsExecutor,
                            # PowerSpectrumAnalysisExecutor,
                            # PowerSpectrumSummaryExecutor,
                            # PowerSpectrumSummaryFigsExecutor,
                            # # ShowOnePSExecutor,
                            # PostAnalysisPsFigExecutor
                            )

logger = logging.getLogger(__name__)

# config_pyilc_t_HILC_backup

@hydra.main(version_base=None, config_path="cfg", config_name="config_pyilc")
def main(cfg):
    logger.debug(f"Running {__name__} in {__file__}")

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    pipeline_context.add_pipe(HydraConfigCheckerExecutor)

    # In the following, "Common" means "Apply the same postprocessing to all models"; requires a mask
    # Apply to the target (CMB realization)
    pipeline_context.add_pipe(CommonRealPostExecutor)
    pipeline_context.add_pipe(CommonPredPostExecutor)

    # # Show results of cleaning
    pipeline_context.add_pipe(CommonShowSimsPostExecutor)  # Temporary fix
    # pipeline_context.add_pipe(CommonNILCShowSimsPostIndivExecutor)  # NOT DONE

    pipeline_context.add_pipe(PixelAnalysisExecutor)
    pipeline_context.add_pipe(PixelSummaryExecutor)
    # pipeline_context.add_pipe(PixelSummaryFigsExecutor)  # NOT DONE

    # # # Not needed in every analysis pipeline, but needed in one
    # pipeline_context.add_pipe(ConvertTheoryPowerSpectrumExecutor)  # NOT DONE
    # pipeline_context.add_pipe(MakeTheoryPSStats)  # NOT DONE

    # # PyILC's Predictions as Power Spectra Anaylsis
    # pipeline_context.add_pipe(PyILCMakePSExecutor)  # NOT DONE
    # pipeline_context.add_pipe(PowerSpectrumAnalysisExecutor)  # NOT DONE
    # pipeline_context.add_pipe(PowerSpectrumSummaryExecutor)  # NOT DONE
    # pipeline_context.add_pipe(PowerSpectrumSummaryFigsExecutor)  # NOT DONE
    # pipeline_context.add_pipe(PostAnalysisPsFigExecutor)  # NOT DONE

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
