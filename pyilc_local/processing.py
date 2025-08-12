import healpy as hp
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
#from cmbml.pyilc_redir import ILCInfo, Wavelets, harmonic_ILC, wavelet_ILC
#Collection of relavent functions to be used in pyilc processing


def read_dict_from_yaml(yaml_file):
    #reads from ymal file
    assert(yaml_file != None)
    with open(yaml_file) as f:
        config = yaml.safe_load(f)
    return config


def get_power(mapp1,mapp2,mask=None,maskfsky=None,lmax=None):
    #Calculate powerspectra with mask
    #mask fsky can be used to give an appropriate sky fraction without multiplying by a mask
    if mask is None:
        mask = np.ones(len(mapp1))
    mean1 = np.sum(mapp1*mask)/np.sum(mask)
    mean2 = np.sum(mapp2*mask)/np.sum(mask)

    if maskfsky is None:
        maskfsky = mask
    fsky = np.sum(maskfsky)/maskfsky.shape[0]
    return hp.anafast(mask*(mapp1-mean1),mask*(mapp2-mean2),lmax=lmax)/fsky

def clearfolder(folder,save_weights=False):
    #Clears a folder of all contents
    for filename in os.listdir(folder):
        first_string = filename[:12]
        if save_weights and first_string == 'CN_weightmap':
            pass
        else:
            file_path = os.path.join(folder, filename)
            os.unlink(file_path)

def conv_to_beam(map,fwhm_new,fwhm_orig = 0,Nside=512,lmax=None):
    #Convolves from a certain beam to a new beam. If fwhm_orig left blank will assumed inititally deconvolved
    if lmax is None:
        lmax = 3*Nside-1
    newbeam = hp.gauss_beam(np.radians(fwhm_new/60),lmax=lmax)
    origbeam = hp.gauss_beam(np.radians(fwhm_orig/60),lmax = lmax)
    alm = hp.map2alm(map,lmax = lmax)
    alm_c = hp.almxfl(alm, newbeam/origbeam)
    map_c = hp.alm2map(alm_c,nside=Nside)
    return map_c

def map_nan(mapp,mask,convert_to = 'nan'):
    #Replaces zeros in mask with np.nan (or vis versa)
    copy_map = np.copy(mapp)
    if convert_to == 'nan':
        for i in range(len(mask)):
            if mask[i] == 0:
                copy_map[i] = np.nan
    elif convert_to == '0':
        for i in range(len(mask)):
            if mask[i] == 0:
                copy_map[i] = 0
    return copy_map

def remove_dipole(mapp, mask=None):
    #Removes_dipole by ignoring masked pixels using map_nan
    if mask is not None:
        mapnan = map_nan(mapp,mask,convert_to='nan')
        mapnan = hp.remove_dipole(mapnan)
        nodipole_map = map_nan(mapnan,mask,convert_to='0')
    else:
        nodipole_map = hp.remove_dipole(mapp)
    return nodipole_map

def mean_error(map1,map2,type = 'sq', mask=None):
    #Caluclates mean sq err between maps
    #Put 'single' for map2 if map1 is the residual
    #sq for mean square error, abs for mean absolute error
    if mask is None:
        mask = np.ones(len(map1))
    if map2 == 'single':
        error = map1
    else:
        error = map1-map2
    meansqerr = 0
    if type == 'sq':
        for i in range(len(map1)):
            meansqerr += error[i]**2*mask[i]
    elif type == 'abs':
        for i in range(len(map1)):
            meansqerr += abs(error[i])*mask[i]
    else:
        raise ValueError("Type must be 'sq' or 'abs'")
    meansqerr = meansqerr/np.sum(mask)

    return float(meansqerr)


def plt_ps(mapp_inp,mask_inp=None,lmax = None, fwhm = 20.6,ylim = [1,8000],logmode = False):
    if type(mapp_inp) == str:
       mapp = hp.read_map(mapp_inp)
    else:
        mapp = mapp_inp
    
    if type(mask_inp) == str:
       mask = hp.read_map(mask_inp)
    else:
        mask = mask_inp

    Nside = hp.npix2nside(len(mapp))
    if lmax == None:
        lmax = 2*Nside
    ps = get_power(mapp,mapp,mask,lmax=lmax)
    beam = hp.sphtfunc.gauss_beam(fwhm*(np.pi/180.0/60.0),lmax=lmax)
    ells = np.arange(lmax+1)
    plt.plot(ells,ells*(ells+1)*ps/(2*np.pi*beam**2))
    plt.ylim(ylim[0],ylim[1])
    if logmode:
        plt.xscale('log')
        plt.yscale('log')

def avgplot(y,n):
    #Rolling average for n ells
    ynew = np.full(len(y),np.nan)
    for i in range(int(n/2),int(len(y)-n/2)):
        av = np.sum(y[int(i-n/2):int(i+n/2)])/n
        ynew[i] = av
    return ynew