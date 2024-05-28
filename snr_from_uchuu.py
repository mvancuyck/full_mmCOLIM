import numpy as np
import matplotlib.pyplot as plt
import pickle
import astropy.units as u
from astropy.io import fits
import scipy.constants as cst
from gen_all_sizes_cubes import *
from astropy.stats import gaussian_fwhm_to_sigma
from fcts import *
from astropy.cosmology import Planck15 as cosmo
from functools import partial
import powspec
import argparse
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler

#/home/mvancuyck/CO_paper/CONCERTO_SENSITIVITY/SIDES_sensitivity_1Ghz_res.txt
def sensitivity(Dnu, nu_obs, dnu, tex_file='/home/mvancuyck/Desktop/co-paper-snr/CONCERTO_SENSITIVITY/SIDES_sensitivity_1Ghz_res.txt', beam_from_file=True, test=False):
    #Load the CONCERTO noise file

    nu_sens, NEFD_mJy_beam, Omega_beam_file = np.loadtxt(tex_file, unpack = True)
    Omega_beam = np.interp(nu_obs.to(u.GHz).value, nu_sens, Omega_beam_file*u.sr)
    #Rescale the sensitivity to the spectral resolution
    NEI_MJy_sr = (1e-9 * NEFD_mJy_beam / (Omega_beam) / Dnu.value) * u.MJy
    #Needed to gen noisy maps
    NEFD_mJy_beam_obs =  np.interp(nu_obs.to(u.GHz).value, nu_sens, NEFD_mJy_beam) * u.mJy / Dnu.value
    NEI_MJy_sr_obs   = np.interp(nu_obs.to(u.GHz).value, nu_sens, NEI_MJy_sr)

    if(test):
        input_spectra = Spectrum1D( flux=NEFD_mJy_beam*u.mJy, spectral_axis=nu_sens*u.GHz)
        '''
        n_slices = np.ceil(Dnu / dnu).value.astype(int)*3
        freqs = np.linspace(nu_obs-n_slices/2*dnu, nu_obs+n_slices/2*dnu, int(n_slices+1))
        '''
        freqs_1channel = np.asarray(((nu_obs-Dnu).value,  nu_obs.value, (nu_obs+Dnu).value))*u.GHz
        fluxc_resample = FluxConservingResampler()
        noise = fluxc_resample(input_spectra, freqs_1channel)  
        noise_for_slice = noise.flux[1]
        print(f'noise from Eq. 1 gives {np.round(NEFD_mJy_beam_obs,1)} and my test gives {np.round(noise_for_slice,1)}')
        #plt.plot(np.ones(len(nu_sens)), nu_sens, 'ob')
        plt.plot(freqs_1channel, np.ones(len(freqs_1channel)), 'og')
        plt.plot(noise.spectral_axis.bin_edges, np.ones(len(noise.spectral_axis.bin_edges)), 'ok')
        plt.plot(nu_sens, np.ones(len(nu_sens)), 'or')
    return NEI_MJy_sr_obs, NEFD_mJy_beam_obs.to(u.Jy), Omega_beam

def my_sensitivity(nu_obs, Delta_nu, dnu, NEFD_dual_band, t_survey, field_size_survey):
    '''
    inspired_from_compute_mapping_speed & Hu et al. 

    nu_obs: [GHz]
    Delta_nu: bandwidth [GHz]
    dnu: absolute spectral resolution [GHz]
    NEFD_dual_band: 99/118 LF/HF [mJy/beam]
    t_survey: total integration time, in hours
    field_size_survey: total surveyed area [deg^2]
    '''
    Omega_beam, fwhm = beam(nu_obs) #PS: pix size is >2 times the fwhm (good)
    t_obs, Omega_pix = t_per_pix(t_survey, field_size_survey)
    NEFD_spec_mJy_beam = NEFD_dual_band * Delta_nu / dnu.value 
    NEI_MJy_sr = 1e-9 * NEFD_spec_mJy_beam / Omega_beam 

    return NEI_MJy_sr, NEFD_spec_mJy_beam, Omega_beam, Omega_pix, t_obs


def t_per_pix(t_survey, field_size_survey, fov_diameter=18.54*u.arcmin, npix_tot=2152):
    fov = np.pi*(fov_diameter**2).to(u.deg**2)
    Omega_pix = fov/npix_tot
    t_obs = t_survey * fov.value / field_size_survey / npix_tot * 3600 #secondes, 
    return t_obs, Omega_pix


def cosmo_distance(z, dnu,  nu_obs, zmin, zmax,):
    dz = dnu * (1+z) / nu_obs
    Dc =  cosmo.comoving_distance(z)/u.rad
    delta_Dc = ( (cst.c*1e-3*u.km/u.s) * (1+z) * dnu / cosmo.H(z) / nu_obs)
    pk_3d_to_2d = 1/(Dc**2*delta_Dc)
    k_3d_to_2d  = Dc/2/np.pi

    Dc_min = cosmo.comoving_distance(zmin)
    Dc_max = cosmo.comoving_distance(zmax)
    full_volume_covered = 4/3*np.pi*(Dc_max**3-Dc_min**3)

    return Dc, delta_Dc, pk_3d_to_2d.to(u.sr/u.Mpc**3), k_3d_to_2d, full_volume_covered


def beam(nu, D=11.5*u.m):

    FWHM = 1.22 * cst.c / nu.to(u.Hz).value / D.value * u.rad
    sigma_beam = FWHM * gaussian_fwhm_to_sigma
    Omega_beam = 2.0 * np.pi * sigma_beam ** 2
    return Omega_beam, FWHM

def volume(omega, Dc, z, nu_obs, dnu, full_volume_at_z=None):
    c = cst.c*1e-3*u.km/u.s
    y = (c / cosmo.H(z)) * (1+z)**2  / (nu_obs.to(u.GHz)*(1+z))
    #If flat sky approximation hold, then compute volume with Eq.8
    if(full_volume_at_z is None): volume = (Dc**2).to(u.Mpc**2/u.sr)*(omega).to('sr') * y * dnu
    #Else, compute volume with Eq.7 if the total comoving volume is provided. 
    else:                         volume = omega.to(u.sr) * full_volume_at_z/4/np.pi/u.sr
    return volume

def sigma(P_A, N_modes, P_B=None, P_AB=None):
    if(P_B is None):  P_B  = P_A
    if(P_AB is None): P_AB = np.sqrt(P_A*P_B)
    sigma = np.sqrt(P_AB**2 + (P_A*P_B) ) / np.sqrt(2*N_modes)
    return sigma

def SNR_fct(signal_3d, sigma_3d):
    SNR = signal_3d/sigma_3d
    SNR_comb = np.abs(np.sqrt(np.sum(SNR**2)))
    return  SNR_comb


def plot_results(title, simu, z, dz, line, t_survey,  SNR_J, SNR_JG, Nmode, k_3d_to_2d, pk_3d_to_2d, 
                 k, pkJ, snJ, sigma_J, pkG, snG, pkJG, snJG, sigma_JG, pkJG_interlopers,  klim, kmax, Pk_noise, pkJ_interlopers  ): 

    int_part, slope = print_scientific_notation( SNR_J )
    snrj = f"Auto spectrum SNR={int_part}"+"$\\times$"+"10^("+f"{slope}"+")"
    int_part, slope = print_scientific_notation( SNR_JG )
    snrjg = f"Cross spectrum SNR={int_part}"+"$\\times$"+"10^("+f"{slope}"+")"

    n=3
    auto_sensitivity_nsigma  = n * Pk_noise / (np.sqrt(Nmode)-n)
    cross_sensitivity_nsigma = n * np.sqrt(pkG/pk_3d_to_2d * (pkJ/pk_3d_to_2d+Pk_noise)) / np.sqrt( 2*Nmode-n**2)
    w = np.where((2*Nmode-n**2)>0)
    
    def g(k_3d_to_2d, x): return (x*u.arcmin**-1).to(u.rad**-1).value / k_3d_to_2d.value
    G = partial( g, k_3d_to_2d )
    def f(pk_3d_to_2d, x): return x / pk_3d_to_2d.value
    F = partial( f, pk_3d_to_2d )

    BS = 6; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); mk = 3; lw=1
    fig, axs = plt.subplots(1,3, figsize=(9,3), dpi=200)  
    ax=axs[0]
    ax.errorbar(k, pkJ, yerr=(sigma_J*pk_3d_to_2d), color='k', ecolor='silver', label=f"Intrinsic {line}@z={z}, dz={dz}" )
    ax.loglog(k, pkJ-snJ, "b", label=f"Clustering of {line}@z={z}" )
    ax.loglog(k[w], auto_sensitivity_nsigma[w]*pk_3d_to_2d, label=f'{n}'+'$\\rm \\sigma$ sensitivity', c='g')
    ax.axhline(snJ.value, k.min().value, 7, linestyle=':', color='k', label='Intrinsic shot noise')
    ax.axhline((Pk_noise*pk_3d_to_2d).value, k.min().value, 7, linestyle='--', color='cyan', label= f'Noise pow. spec, t={t_survey}h.')
    ax.axvline(kmax.value, 1e-3, 1e2, linestyle=':', color='brown', )
    ax.axvline(klim.value, 1e-3, 1e2, linestyle=':', color='magenta', )
    ax.loglog(k, pkJ_interlopers,  "r", label=f"{line}@z={z}+Interlopers")
    ax.legend()
    ax.set_title(f"auto power spectrum \n of {simu} in 2D. \n {snrj}. ")
    ax.set_xlabel('k $\\rm [arcmin^{-1}]$')
    ax.set_ylabel('$\\rm P_J$(k) [$\\rm Jy^2/sr$]')
    secax = ax.secondary_xaxis("top", functions=(G,G))
    secax.set_xlabel('k [$\\rm Mpc^{-1}$]')
    secax = ax.secondary_yaxis("right", functions=(F,F))
    secax.set_ylabel('$\\rm P_{J}$(k) [$\\rm Jy^2/sr^2.Mpc^3]$]')
    ax=axs[1]
    ax.loglog(k, pkG, "k")
    ax.axhline(snG.value, k.min().value, 7, linestyle=':', color='k', label='Intrinsic shot noise')
    ax.loglog(k, pkG-snG, "b", label=f"Clustering")
    ax.set_title(f"auto power spectrum \n of {simu} galaxies at z={z} , dz={dz} in 2D.")
    ax.set_xlabel('k $\\rm [arcmin^{-1}]$')
    ax.set_ylabel('$\\rm P_G$(k) [$\\rm sr$]')
    secax = ax.secondary_xaxis("top", functions=(G,G))
    secax.set_xlabel('k [$\\rm Mpc^{-1}$]')
    secax = ax.secondary_yaxis("right", functions=(F,F))
    secax.set_ylabel('$\\rm P_{G}$(k) [$\\rm Mpc^3]$')
    ax=axs[2]
    ax.errorbar(k, pkJG, yerr=(sigma_JG*pk_3d_to_2d), color='k', ecolor='silver', label=f"Intrinsic Gal-{line}@z={z}")
    #ax.loglog(k, pkJG-snJG, "b", label=f"Clustering of Gal-{line}@z={z}, dz={dz}" )
    ax.loglog(k, pkJG_interlopers, "r", label=f"Gal-{line}@z={z}+interlopers")
    ax.axhline(snJG.value, k.min().value, 7, linestyle=':', color='k', label='Intrinsic shot noise')
    ax.loglog(k[w], cross_sensitivity_nsigma[w]*pk_3d_to_2d, label=f'{n}'+'$\\rm \\sigma$ sensitivity', c='g')
    #ax.axvline(klim.value, 1e-3, 1e2, linestyle=':', color='magenta')
    ax.axvline(kmax.value, 1e-3, 1e2, linestyle=':', color='brown', label = 'CONCERTO $\\rm k_{max}$')
    ax.axvline(klim.value, 1e-3, 1e2, linestyle=':', color='magenta', label = '$\\rm k_{min}$ in SNR')
    ax.legend()
    ax.set_title(f"cross power spectrum \n of {simu} at z={z} in 2D.  \n {snrjg}. ")
    ax.set_xlabel('k $\\rm [arcmin^{-1}]$')
    ax.set_ylabel('$\\rm P_{J\\times G}$(k) [$\\rm Jy$]')
    secax = ax.secondary_xaxis("top", functions=(G,G))
    secax.set_xlabel('k [$\\rm Mpc^{-1}$]')
    secax = ax.secondary_yaxis("right", functions=(F,F))
    secax.set_ylabel('$\\rm P_{J\\times G}$(k) [$\\rm Jy.sr^{-1}.Mpc^3]$')
    #ax.set_yscale('linear')
    fig.tight_layout()
    fig.savefig(f'{title}.pdf', transparent=True)
    fig.savefig(f'{title}.png', transparent=True)

    return 0

def main_paper(simu, cat, simu_field_size, field_size_survey, line, rest_freq, z, dz, t_survey, npix, 
               dtype="all_lines", n_slices=0.0, dnu = 1.5*u.GHz, plot_map=False):


    #Load the dicts for transition of reference 
    file = f"dict_dir/dict_LIMgal_{simu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{simu_field_size}deg2_{line}{dtype}.p"
    dict= pickle.load( open(file, 'rb'))
    file = f"dict_dir/dict_LIMgal_{simu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{simu_field_size}deg2_{line}.p"
    d = pickle.load( open(file, 'rb'))

    #-----------------             
    #Without interlopers
    if(plot_map): 
        hdul = fits.open(f'{output_path}/{simu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{simu_field_size}deg2_{line}_MJy_sr.fits')
        data_cube = hdul[0].data  
        wcs_cube = wcs.WCS(hdul[0].header)
        data_slice = np.abs(data_cube[0, :, :])+1e-5
        wcs_slice = wcs_cube.dropaxis(2)    
        fig, ax = plt.subplots(subplot_kw={'projection': wcs_slice}, dpi=200, figsize=(4.5,4.5))
        # Plot the slice
        ax.set_title(f"{line} at z={z}")
        im = ax.imshow(data_slice, cmap='viridis', origin='lower', aspect='auto', norm=LogNorm())
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label='Intensity [MJy$\\rm sr^{-1}$]')
        # Add labels to the axes with WCS coordinates
        ax.set_xlabel('RA (deg)')
        ax.set_ylabel('Dec (deg)')
        #fig.tight_layout()
    #-----------------
    
    zmin = z-dz/2 ##!!
    zmax = z+dz/2 ##!!
    freq_obs = rest_freq/(1+z)
    Dnu = dz * freq_obs/(1+z) ##!!
    NEI_MJy_sr, NEFD_Jy_beam, Omega_beam = sensitivity(Dnu, freq_obs, dnu) ##!!
    Dc, delta_Dc, pk_3d_to_2d, k_3d_to_2d, full_volume_covered = cosmo_distance(z, Dnu, freq_obs, zmin, zmax)##!!
    #
    Vpix    = volume(Omega_beam, Dc, z, freq_obs, Dnu) ##!!
    Vsurvey = field_size_survey.to(u.sr) * full_volume_covered / (4*np.pi)
    t_obs = t_survey *  npix * Omega_beam.to(u.deg**2) / field_size_survey.to(u.deg**2)
    Pk_noise = (Vpix * NEI_MJy_sr**2/ t_obs).to(u.Jy**2/u.sr**2*u.Mpc**3) 

    pkJ =   np.abs(d['pk_J_0'])
    pkG =         (d['pk_gal_0']*u.sr**2).to(u.sr)
    pkJG = (np.abs(d['pk_J-gal_0']) * u.sr).to(u.Jy)
    pkJ_3d  = (pkJ   / pk_3d_to_2d).to(u.Jy**2/u.sr**2*u.Mpc**3)
    pkG_3d  = (pkG   / pk_3d_to_2d).to(u.Mpc**3)
    pkJG_3d = (pkJG  / pk_3d_to_2d).to(u.Jy/u.sr*u.Mpc**3)
    
    snJ = d['LIM_shot_list'][0]
    snG = d['gal_shot_list'][0]*u.sr
    snJG = (d['LIMgal_shot_list'][0]  * u.sr).to(u.Jy)
    snJ_3d  = (snJ   / pk_3d_to_2d).to(u.Jy**2/u.sr**2*u.Mpc**3)
    snG_3d  = (snG   / pk_3d_to_2d).to(u.Mpc**3)
    snJG_3d  = (snJG / pk_3d_to_2d).to(u.Jy/u.sr*u.Mpc**3)
    
    res = d['res']
    k   = d['k']#.to(u.rad**-1)
    K   = k.to(u.rad**-1)/k_3d_to_2d #Mpc^-1
    delta_k = np.diff(d["kbin"]) ##!!
    delta_K = delta_k.to(u.rad**-1)/k_3d_to_2d #Mpc^-1
    Nmode = 2 * np.pi * (K**2 * delta_K) * Vsurvey/ (2 * np.pi)**3
    kmax = (1/np.sqrt(field_size_survey).to(u.rad)).to(1/u.arcmin)
    Klim = (1.8e-1/u.Mpc) #1.35
    klim = (Klim*k_3d_to_2d).to(1/u.arcmin)

    w = np.where(k<klim)[0][-1]+1
    sigma_J  = sigma(pkJ_3d+Pk_noise,  Nmode)
    sigma_G  = sigma(pkG_3d,           Nmode)
    sigma_JG = sigma(pkJ_3d+Pk_noise,  Nmode, pkG_3d, pkJG_3d)

    SNR_J = SNR_fct(pkJ_3d[:w], sigma_J[:w])
    SNR_G = SNR_fct(pkG_3d[:w], sigma_G[:w])

    SNR_JG = SNR_fct(pkJG_3d[:w], sigma_JG[:w])

    plot_results(f"{simu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{simu_field_size}deg2_{line}", 
                simu, z, dz, line, t_survey, SNR_J, SNR_JG, Nmode,
                k_3d_to_2d, pk_3d_to_2d, 
                k, pkJ, snJ, sigma_J, pkG, snG, pkJG, snJG, sigma_JG, pkJG_interlopers,  
                klim, kmax, Pk_noise, pkJ_interlopers)

    return k, pkJ, snJ, pkG, snG, pkJG, snJG
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="gen cubes from Uchuu",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('output_path', help="output path of products", default = '.')
    args = parser.parse_args()
    #Load the output path
    output_path = args.output_path

    #need to load the full Uchuu catalogue to compute the shot noises. 
    #Otherwise, load directly the results in the .p file, 
    cat=None
    #_, cat, _, _ = load_cat()

    #With SIDES Bolshoi, for rapid tests. 
    tim_params = load_params('PAR/cubes.par')
    z_list = tim_params['z_list']
    dz_list = tim_params['dz_list']
    n_list = tim_params['n_list']

    CONCERTO = {'npix':int(2152*0.725),
               'Omega_field':1.5*u.deg**2,
               't_survey': 600,
               'spectral_resolution':1.5,
               'FWHM':30,
                }

    for nslice, dz in zip(n_list,dz_list): 
        
        dict = {}
        for line, rest_freq in zip(line_list[2:5],rest_freq_list[2:5]):
            k, pkJ, snJ, pkG, snG, pkJG, snJG = main_paper('pySIDES_from_uchuu', cat, 117, 
                                                           CONCERTO['Omega_field'], line, rest_freq, 
                                                           1.0, dz, CONCERTO['t_survey'], 
                                                           CONCERTO['npix'], n_slices=nslice, 
                                                           dnu=CONCERTO['spectral_resolution'])

        dict = {}
        for line, rest_freq in zip(line_list[2:4],rest_freq_list[2:4]):        
            k, pkJ, snJ, pkG, snG, pkJG, snJG = main_paper('pySIDES_from_uchuu', cat, 117, 
                                                           CONCERTO['Omega_field'], line, rest_freq, 
                                                           1.5, dz, CONCERTO['t_survey'], CONCERTO['npix'], 
                                                           n_slices=nslice, dnu=CONCERTO['spectral_resolution'])
        
    plt.show()
    









    '''
            dict[f'pk_auto_{line}_tot'] = pkJ
            dict[f'auto_shot_noise_{line}']  = snJ
            dict[f'pk_cross_gal-{line}_tot'] = pkG
            dict[f'cross_shot_noise_gal-{line}'] = snJG
        dict[f'pk_auto_galaxies_tot'] = pkG
        dict[f'auto_shot_noise_galaxies'] = snG
        dict['k_per_arcmin'] = k
        pickle.dump(dict, open(f'power_spectra_in_pySIDES_from_uchuu_50arcsec_resolution_at_z1.0_in_1slice_of_dz{dz}_no_interlopers.p', 'wb'))
    '''