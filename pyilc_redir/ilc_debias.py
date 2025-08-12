from __future__ import print_function
import sys
#from typing import Self
import numpy as np
import os
import healpy as hp
import pyilc_local.processing as pros
from . import ILCInfo, Wavelets, harmonic_ILC, wavelet_ILC
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False
from pyilc_redir.pyilc_wrapper import run_ilc
import copy
import yaml


#pyilc_folder = '/bigdata/seth/ml-model/pyilc-main/'

class Pyilc_Debias:
    def __init__(self,cfg_input):

        self.cfg_input = cfg_input

        self.datasetdir = f"{os.getenv('CMB_ML_DATA')}/Datasets/{self.cfg_input['dataset_name']}/"

        self.datasetfolder = 'Simulation_Half/Test/'

        self.sim_number = self.cfg_input['simulation']
        
        #Will automatically run half mission soon
        if not os.path.exists(self.datasetdir+self.datasetfolder):
            raise FileNotFoundError('Need to run half_mission_construction!')

        self.working_folder = f"{self.datasetdir}Pyilc_Working/"
        if not os.path.exists(self.working_folder):
            print("Creating Working Folder")
            os.makedirs(self.working_folder)

        self.output_folder = f"{self.datasetdir}Pyilc_Output_{self.sim_number}a/"
        if not os.path.exists(self.output_folder):
            print("Creating Output Folder")
            os.makedirs(self.output_folder)

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
        self.beam_smooth_mask = pros.conv_to_beam(self.mask,self.fwhm,Nside=self.Nside)


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


    def run_specific_sim(self,sim,input_type,simulation_folder='/Simulation/Test/',simulation_dir=None):
        #if cleaned output for a simulation doesn't exist, preform cleaning and create one
        self.input_type= input_type

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
        print('CLEAN!!:',clean)

        hp.fitsfunc.write_map(f"{self.outputsim_folder}{self.input_type}.fits",clean,overwrite=True)

        #weights are saved to be used later
        pros.clearfolder(self.working_folder,save_weights=True)

        return clean



    def create_weighted_noise(self,noise_dir):
        #applies previous weights to each noise map template in debiasing_simulations/noise
        print('Creating Weighted Noise Maps')

        self.noise_map_vector = []

        input_type = "noise"
        for sim in os.listdir(noise_dir):
            weighted_noise_map = self.run_specific_sim(sim,input_type,simulation_dir=noise_dir)
            self.noise_map_vector.append(weighted_noise_map)



    def cross_spectra_estimate(self):
        #calculates and averages all possible cross spectra between weighted noise maps
        print('Estimating Cross Spectra')
        cross_ps_list = []
        cross_spec_dir = f"{self.output_folder}Noise_Cross_PS.txt"
        if not os.path.exists(cross_spec_dir):
            for i in range(len(self.noise_map_vector)):
                for j in range(i + 1, len(self.noise_map_vector)):
                    print('Computing Cross Spectra Between',i,'and',j)
                    cross_ps = pros.get_power(self.noise_map_vector[i],self.noise_map_vector[j],maskfsky=self.beam_smooth_mask,lmax=self.lmax)
                    cross_ps_list.append(cross_ps)
            cross_ps_list = np.array(cross_ps_list)
            self.avg_cross_spec = np.mean(cross_ps_list,axis=0)
            np.savetxt(cross_spec_dir,self.avg_cross_spec)
        else:
            self.avg_cross_spec = np.loadtxt(cross_spec_dir)


    def ps_analysis(self, clean_map, resid_map, fg_map):
        #calculates power spectra and preforms proper corrections

        if self.cfg_input['type_of_map_2b_cleaned'] == 'half':
            n = 2
        else:
            n = 4
        cleanps = pros.get_power(clean_map,clean_map,maskfsky=self.beam_smooth_mask,lmax=self.lmax)
        residps = pros.get_power(resid_map,resid_map,maskfsky=self.beam_smooth_mask,lmax=self.lmax)
        fidps = pros.get_power(self.fid_map,self.fid_map,lmax=self.lmax)
        fgps = pros.get_power(fg_map,fg_map,maskfsky=self.beam_smooth_mask,lmax=self.lmax)
        crossps = self.avg_cross_spec
        correctedps = cleanps - residps/n - 2*crossps/n - fgps

        beam = hp.sphtfunc.gauss_beam(self.fwhm*(np.pi/180.0/60.0),lmax=self.lmax)
        ells = np.arange(self.lmax+1)

        correctedps_D_deconv = ells*(ells+1)*correctedps/(2*np.pi*beam**2)
        fidps_D = ells*(ells+1)*fidps/(2*np.pi)
        avg_correctedps = pros.avgplot(correctedps_D_deconv,34)

        np.savetxt(f"{self.output_folder}Estimated_PS.txt",correctedps_D_deconv)
        np.savetxt(f"{self.output_folder}Fiducial_PS.txt",fidps_D)


        #plotting
        plt.figure()
        plt.plot(ells,correctedps_D_deconv,label = 'Estimated Power Spectrum')
        plt.plot(ells,fidps_D,label='Fiducial Power Spectrum')
        #plt.plot(ells,ells*(ells+1)/(2*np.pi*beam**2)*cleanps,label = 'Clean Power Spectrum')
        plt.plot(ells,avg_correctedps,'k',linewidth=1,label = 'Avg Estimated Power Spectrum')

        plt.xlabel('$\ell$')
        plt.ylabel('$D_\ell$')
        plt.ylim(0,8000)
        plt.title(f"Power Spectrum Estimation")
        plt.legend()
        plt.savefig(f"{self.output_folder}plotted_cmb_ps.png")



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






#Runs Cleaning + Full Debias

def pyilc_debias_main(cfg_input):
    sim_number = cfg_input['simulation']
    obs_sim = sim_number+'a'

    ilc_init = Pyilc_Debias(cfg_input)

    noise_dir = f"{os.getenv('CMB_ML_DATA')}/Assets/debiasing_simulations/noise/"
    foreground_dir = f"{os.getenv('CMB_ML_DATA')}/Assets/debiasing_simulations/foreground/"


    obs_map = ilc_init.run_specific_sim(obs_sim,'obs',simulation_folder=ilc_init.datasetfolder)/1e6

    resid_sim = ilc_init.prepare_residuals()

    resid_map = ilc_init.run_specific_sim(resid_sim,'resid',simulation_folder=ilc_init.datasetfolder)/1e6

    fg_map = ilc_init.run_specific_sim('fg0','fg',simulation_dir=foreground_dir)/1e6

    ilc_init.create_weighted_noise(noise_dir)

    ilc_init.cross_spectra_estimate()

    ilc_init.ps_analysis(obs_map,resid_map,fg_map)

    pros.clearfolder(ilc_init.working_folder,save_weights=False)