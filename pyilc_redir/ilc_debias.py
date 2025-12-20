from __future__ import print_function
import sys
#from typing import Self
import numpy as np
import os
import healpy as hp
from . import ILCInfo, Wavelets, harmonic_ILC, wavelet_ILC
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
from pyilc_redir.pyilc_wrapper import run_ilc
import copy
import yaml


#These functions are in the processing file, but I've copied them here to it's clear what they do

def conv_to_beam(map,fwhm_new,fwhm_orig = 0,lmax=None):
    #Convolves from a certain beam to a new beam. If fwhm_orig left blank will assumed inititally deconvolved
    Nside = hp.get_nside(map)
    if lmax is None:
        lmax = 3*Nside-1
    newbeam = hp.gauss_beam(np.radians(fwhm_new/60),lmax=lmax)
    origbeam = hp.gauss_beam(np.radians(fwhm_orig/60),lmax = lmax)
    alm = hp.map2alm(map,lmax = lmax)
    alm_c = hp.almxfl(alm, newbeam/origbeam)
    map_c = hp.alm2map(alm_c,nside=Nside)
    return map_c

def get_power(mapp1,mapp2,mask=None,maskfsky=None,lmax=None,iter=3):
    #Calculate powerspectra with mask
    #mask fsky can be used to give an appropriate sky fraction without multiplying by a mask
    if mask is None:
        mask = np.ones(len(mapp1))
    mean1 = np.sum(mapp1*mask)/np.sum(mask)
    mean2 = np.sum(mapp2*mask)/np.sum(mask)

    if maskfsky is None:
        maskfsky = mask
    fsky = np.sum(maskfsky**2)/maskfsky.shape[0]
    return hp.anafast(mask*(mapp1-mean1),mask*(mapp2-mean2),lmax=lmax,iter=iter)/fsky

def clearfolder(folder,save_weights=False):
    #Clears a folder of all contents
    for filename in os.listdir(folder):
        first_string = filename[:12]
        if save_weights and first_string == 'CN_weightmap':
            pass
        else:
            file_path = os.path.join(folder, filename)
            os.unlink(file_path)




#pyilc_folder = '/bigdata/seth/ml-model/pyilc-main/'

class Pyilc_Debias:
    def __init__(self,cfg_input):

        self.cfg_input = cfg_input

        self.datasetdir = f"{os.getenv('CMB_ML_DATA')}/Datasets/{self.cfg_input['dataset_name']}/"

        self.datasetfolder = 'Simulation_Half/Test/'

        self.sim_number = self.cfg_input['simulation']
        
        working_dir = self.cfg_input['working_dir']

        if not os.path.exists(self.datasetdir+self.datasetfolder):
            raise FileNotFoundError('Need to run half_mission_construction!')

        self.working_folder = f"{self.datasetdir}{working_dir}NILC_A_Working_Dir/"
        if not os.path.exists(self.working_folder):
            print("Creating Working Folder")
            os.makedirs(self.working_folder)

        self.output_folder = f"{self.datasetdir}{working_dir}NILC_C_Debias/{self.sim_number}/"
        if not os.path.exists(self.output_folder):
            print("Creating Output Folder")
            os.makedirs(self.output_folder)

        #Need to read units from map metadata soon. Can only take in K currently

        # if 'units' in self.cfg_input.keys():
        #     if self.cfg_input['units'] == 'K':
        #         self.unit_factor = 1e6
        #     else:
        #         self.unit_factor = 1

        self.fid_map = hp.read_map(f"{self.datasetdir}{self.datasetfolder}{self.sim_number}a/cmb_map_fid.fits")


        self.fwhm = self.cfg_input['perform_ILC_at_beam']
        self.Nside = self.cfg_input['N_side']
        self.lmax = self.cfg_input['ELLMAX']

        #Avoiding escaped brakets problem with reading from config_all_dets
        self.mask_path = f'{self.datasetdir}Simulation_Mask/mask.fits'
        #mask_path = self.cfg_input['mask_before_covariance_computation'][0]
        self.mask = hp.read_map(self.mask_path)
        self.apo_mask = conv_to_beam(self.mask,200)


    def run_specific_sim(self,sim,input_type,simulation_folder='/Simulation/Test/',simulation_dir=None):
        #if cleaned output for a simulation doesn't exist, preform cleaning and create one
        self.input_type = input_type

        if simulation_dir is None:
            self.datasimfolder = self.datasetdir+simulation_folder+sim+'/'
        else:
            self.datasimfolder = simulation_dir+sim+'/'

        self.outputsim_folder = f"{self.output_folder}{sim}/"
    
        print(self.outputsim_folder)
        if not os.path.exists(self.outputsim_folder):
            os.makedirs(self.outputsim_folder)
            self.sim_cfg = self.config_creator(input_type)
            pyilc_output = self.run_pyilc()
        else:
            print('Cleaned Simulation Already Exists for: ',sim, ' Skipping.')
            pyilc_output = hp.read_map(f'{self.outputsim_folder}{self.input_type}.fits')

        return pyilc_output
    

    def run_pyilc(self):
        #preform cleaning
        print('Begin cleaning')

        clean = run_ilc(self.sim_cfg)

        hp.fitsfunc.write_map(f"{self.outputsim_folder}{self.input_type}.fits",clean,overwrite=True)

        #weights are saved to be used later
        clearfolder(self.working_folder,save_weights=True)

        return clean
    

    def ps_analysis(self, clean_map, resid_map, fg_map=None, halfa_map=None, halfb_map=None, iter=3):
        #calculates power spectra and preforms proper corrections

        print('Preforming Analysis of Power Spectra')
        cleanps = get_power(clean_map,clean_map,maskfsky=self.apo_mask,lmax=self.lmax,iter=iter)
        residps = get_power(resid_map,resid_map,maskfsky=self.apo_mask,lmax=self.lmax,iter=iter)
        fidps = get_power(self.fid_map,self.fid_map,mask=self.apo_mask,lmax=self.lmax,iter=iter)
        half_crossps = get_power(halfa_map,halfb_map,mask=self.apo_mask,lmax=self.lmax,iter=iter)
        fgps = get_power(fg_map,fg_map,maskfsky=self.apo_mask,lmax=self.lmax,iter=iter)

        #Cross spectra method or residual method
        corrected_ps = half_crossps-fgps
        #corrected_ps = cleanps - residps/4 - fgps

        beam = hp.sphtfunc.gauss_beam(self.fwhm*(np.pi/180.0/60.0),lmax=self.lmax)
        ells = np.arange(self.lmax+1)

        fid_D = ells*(ells+1)*fidps/(2*np.pi)

        corrected_D_deconv = ells*(ells+1)*corrected_ps/(2*np.pi*beam**2)

        np.savetxt(f"{self.output_folder}Estimated_PS.txt",corrected_D_deconv)
        np.savetxt(f"{self.output_folder}Fiducial_PS.txt",fid_D)

        #plotting
        plt.figure()
        plt.plot(ells,fid_D,label='Fiducial Power Spectrum')
        plt.plot(ells,corrected_D_deconv,label = 'Cleaned and Debiased Estimate')
    
        plt.xlabel('$\ell$')
        plt.ylabel('$D_\ell$')
        #plt.ylim(1050,1800)
        #plt.xlim(920,1024)
        plt.ylim(0,6500)
        plt.title(f"Power Spectrum Estimation")
        plt.legend()
        plt.savefig(f"{self.output_folder}plotted_cmb_ps.svg")

    def config_creator(self,input_type):
        #creates config for each individual sim
        base_config = self.cfg_input
        new_config = copy.deepcopy(self.cfg_input)

        new_beams = []
        for fwhm in self.cfg_input['beam_FWHM_arcmin']:
            new_beams.append(float(fwhm))
        new_config['beam_FWHM_arcmin'] = new_beams

        #Fill in each value for run
        ellpeaks = base_config['ellpeaks']
        freqs = base_config['freqs_delta_ghz']

        #Fill in values that are forced
        new_config['ELLMAX'] = ellpeaks[-1] - 1
        new_config['N_scales'] = len(ellpeaks)+1
        new_config['N_freqs'] = len(freqs)
        new_config['output_dir'] = self.working_folder
        new_config['mask_before_covariance_computation'] = [self.mask_path,0]

        
        freq_map_files = [None]*len(freqs)

        for i in range(len(freqs)):
            freq_map_files[i] = f"{self.datasimfolder}{input_type}_{freqs[i]}_map.fits"

        new_config['freq_map_files'] = freq_map_files

        filename = f"config_{input_type}.yaml"
        config_path = f"{self.outputsim_folder}{filename}"
        with open(config_path, "w") as f:
                yaml.dump(new_config, f)
        return config_path


    #Not needed if we only use cross spectra method

    def prepare_residuals(self):
        #creates a residual map (noise estimate) from half mission maps
        half1_folder = self.sim_number+'a'
        half2_folder = self.sim_number+'b'

        resid_folder = f'resid_{half1_folder}-{half2_folder}'

        resid_path = f'{self.datasetdir}{self.datasetfolder}{resid_folder}/'

        if not os.path.exists(resid_path):
            print("Creating Residuals Folder")
            os.makedirs(resid_path)

        freqs = self.cfg_input['freqs_delta_ghz']
        for freq in freqs:
            half1 = hp.read_map(f'{self.datasetdir}{self.datasetfolder}{half1_folder}/obs_{freq}_map.fits')
            half2 = hp.read_map(f'{self.datasetdir}{self.datasetfolder}{half2_folder}/obs_{freq}_map.fits')
            resid = half2-half1
            hp.write_map(f'{resid_path}resid_{freq}_map.fits',resid,overwrite=True)
        return resid_folder




#Runs Cleaning + Full Debias

def pyilc_debias_main(cfg_input):
    sim_number = cfg_input['simulation']

    ilc_init = Pyilc_Debias(cfg_input)

    foreground_dir = f"{os.getenv('CMB_ML_DATA')}/Assets/debiasing_simulations/foreground/"


    obs_map = ilc_init.run_specific_sim(sim_number,'obs',simulation_folder='/Simulation/Test/')

    ###
    #Not needed for cross spectra method
    resid_sim = ilc_init.prepare_residuals()

    resid_map = ilc_init.run_specific_sim(resid_sim,'resid',simulation_folder=ilc_init.datasetfolder)
    #####
    
    halfa_map = ilc_init.run_specific_sim(sim_number+'a','obs',simulation_folder=ilc_init.datasetfolder)
    halfb_map = ilc_init.run_specific_sim(sim_number+'b','obs',simulation_folder=ilc_init.datasetfolder)

    fg_map = ilc_init.run_specific_sim('fg0','fg',simulation_dir=foreground_dir)
  
    ilc_init.ps_analysis(obs_map,resid_map,fg_map,halfa_map,halfb_map)

    clearfolder(ilc_init.working_folder,save_weights=False)