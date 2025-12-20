import os
import yaml
import logging
import hydra
import shutil
import pyilc_local.processing as pros
import healpy as hp
from omegaconf import OmegaConf
from cmbml.core import PipelineContext, LogMaker
from cmbml.core.A_check_hydra_configs import HydraConfigCheckerExecutor
from cmbml.sims import (
    HydraConfigSimsCheckerExecutor,
    NoiseCacheExecutor,
    GetPlanckNoiseSimsExecutor,
    MakePlanckAverageNoiseExecutor,
    MakePlanckNoiseModelExecutor,
    DownloadNoiseModelExecutor,
    ChainsConfigExecutor,
    ParamConfigExecutor,
    TheoryPSExecutor,
    ObsCreatorExecutor,
    PrepForegroundsExecutor,
    NoiseMapCreatorExecutor,
    SimCreatorExecutor
)

logger = logging.getLogger(__name__)

config_path = '/bigdata/seth/ml-model/cmb-ml-pyilc/cfg'

@hydra.main(version_base=None, config_path=config_path,config_name = 'cfg_debias_sims')
def run_simulations(cfg):
    """
    Runs the simulation pipeline.

    Args:
        cfg: The configuration object.

    Raises:
        Exception: If an exception occurs during the pipeline execution.
    """
    logger.debug(f"Running {__name__} in {__file__}")

    print(OmegaConf.to_yaml(cfg))

    log_maker = LogMaker(cfg)
    log_maker.log_procedure_to_hydra(source_script=__file__)

    pipeline_context = PipelineContext(cfg, log_maker)

    # pipeline_context.add_pipe(HydraConfigCheckerExecutor)
    # pipeline_context.add_pipe(HydraConfigSimsCheckerExecutor)

    # Required for the kinds of noise implemented in the pipeline
    pipeline_context.add_pipe(NoiseCacheExecutor)

    pipeline_context.add_pipe(DownloadNoiseModelExecutor)

    ############################
    # Simulation creation
    ############################

    # Needed for all:
    if cfg.model.sim.cmb.use_chains:
        pipeline_context.add_pipe(ChainsConfigExecutor)
    else:
        pipeline_context.add_pipe(ParamConfigExecutor)

    pipeline_context.add_pipe(TheoryPSExecutor)
    pipeline_context.add_pipe(ObsCreatorExecutor)
    pipeline_context.add_pipe(NoiseMapCreatorExecutor)

    # # TODO: Put this back in the pipeline yaml; fix/make executor
    # # pipeline_context.add_pipe(ShowSimsExecutor)  # Out of date, do not use.

    pipeline_context.prerun_pipeline()

    had_exception = False
    try:
        pipeline_context.run_pipeline()
    except Exception as e:
        had_exception = True
        logger.exception("An exception occurred during the pipeline.", exc_info=e)
        raise e
    finally:
        if had_exception:
            logger.error("Pipeline failed.")
        else:
            logger.info("Pipeline completed.")
        log_maker.copy_hydra_run_to_dataset_log()



#if __name__ == "__main__":
  #  run_simulations()


config_dir = f"{config_path}/cfg_debias_sims.yaml"

with open(config_dir) as f:
    config = yaml.safe_load(f)

totalfreqs = [30,44,70,100,143,217,353,545,857]
fwhms = [75,65,55,43,32.4,22.3,22.0,21.5,20.6]  #read from table later

beamdict = dict(zip(totalfreqs,fwhms))

freqs = config['detectors']
Nside = config['nside']

fgassetdir = f"{os.getenv('CMB_ML_DATA')}/Assets/debiasing_simulations/foreground"
noiseassetdir = f"{os.getenv('CMB_ML_DATA')}/Assets/debiasing_simulations/noise"

if not os.path.exists(fgassetdir):
    os.makedirs(fgassetdir)

if not os.path.exists(noiseassetdir):
    os.makedirs(noiseassetdir)

#generalize i51211DB!!!!
datasetfolder = f"{os.getenv('CMB_ML_DATA')}/Datasets/I_512_1-1_DB_obs/Simulation_Working"

emissiondir = f"{datasetfolder}/Simulation_D_Sky_Emission/Test"
noisedir = f"{datasetfolder}/Simulation_E_Noise/Test"


for sim in os.listdir(noisedir):
    print('Sim',sim) 
    shutil.move(f"{noisedir}/{sim}",f"{noiseassetdir}/{sim}")
    

sim = 'sim0000'

cmb = hp.read_map(f'{emissiondir}/{sim}/cmb_map_fid.fits')

if not os.path.exists(f"{fgassetdir}/{sim}"):
    os.makedirs(f"{fgassetdir}/{sim}")
for freq in freqs:
        fwhm = beamdict[freq]
        emission = hp.read_map(f'{emissiondir}/{sim}/sky_{freq}_no_noise_map.fits')
        fg = emission - pros.conv_to_beam(cmb,fwhm,Nside=Nside)
        hp.write_map(f"{fgassetdir}/{sim}/fg_{freq}_map.fits",fg,overwrite=True)