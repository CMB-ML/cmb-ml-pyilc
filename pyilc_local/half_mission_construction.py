import os
import healpy as hp
import processing as pros
import random
#work in progress script to turn one simulation into two half mission simulations provided there are noise only simulations.

##Make sure datasetdir of choice has a working folder with both Sky Emission map and Noise only maps.

dataset_name = 'I_512_1-10'

datasetdir = f"{os.getenv('CMB_ML_DATA')}/Datasets/{dataset_name}/"


workingdir = 'Simulation_Working'
signal_dir = datasetdir+workingdir+'/Simulation_D_Sky_Emission/Test/'
noise_dir = datasetdir+workingdir+'/Simulation_E_Noise/Test/'
output_dir = datasetdir+'/Simulation_Half/Test/'

if not os.path.exists(output_dir):
     print('Creating Output Folder')
     os.makedirs(output_dir)

signalfiles = os.listdir(signal_dir)
noisefiles1 = os.listdir(noise_dir)
noisefiles2 = os.listdir(noise_dir)

n = len(signalfiles)

if n < 2:
    raise ValueError('Too few simulations in dataset.')

#For each signal, two random noise maps are given. In any given split, the same noise map will be used twice
random.shuffle(noisefiles1)
random.shuffle(noisefiles2)
shuffled_noise = noisefiles1 + noisefiles2

print(shuffled_noise)
if shuffled_noise[n-1] == shuffled_noise[n]:
    temp = shuffled_noise[n]
    shuffled_noise[n] = shuffled_noise[n+1]
    shuffled_noise[n+1] = temp

freqs = [44,70,100,143,217,353,545]

for i in range(len(signalfiles)):

    sim = signalfiles[i]
    sim0 = shuffled_noise[i*2]
    sim1 = shuffled_noise[i*2+1]

    for half in ['a','b']:
        output_file = f"{output_dir}{sim}{half}/"
        if not os.path.exists(output_file):
                print("Creating Simulation Output Folder")
                os.makedirs(output_file)

    fid = hp.read_map(f'{signal_dir}{sim}/cmb_map_fid.fits')

    hp.write_map(f'{output_dir}{sim}a/cmb_map_fid.fits',fid,overwrite=True)
    hp.write_map(f'{output_dir}{sim}b/cmb_map_fid.fits', fid,overwrite=True)


    for freq in freqs:
        signal = hp.read_map(f"{signal_dir}{sim}/sky_{freq}_no_noise_map.fits")
        noise0 = hp.read_map(f"{noise_dir}{sim0}/noise_{freq}_map.fits")
        noise1 = hp.read_map(f"{noise_dir}{sim1}/noise_{freq}_map.fits")

        obs0 = signal+noise0
        obs1 = signal+noise1
        hp.write_map(f'{output_dir}{sim}a/obs_{freq}_map.fits',obs0,overwrite=True)
        hp.write_map(f'{output_dir}{sim}b/obs_{freq}_map.fits', obs1,overwrite=True)