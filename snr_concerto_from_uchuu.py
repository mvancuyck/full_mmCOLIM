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

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers

line_list_fancy = ["CO({}-{})".format(J_up, J_up - 1) for J_up in range(1, 9)]

#/home/mvancuyck/CO_paper/CONCERTO_SENSITIVITY/SIDES_sensitivity_1Ghz_res.txt
def sensitivity(dnu, nu_obs, tex_file='/home/mvancuyck/Desktop/co-paper-snr/CONCERTO_SENSITIVITY/SIDES_sensitivity_1Ghz_res.txt', beam_from_file=True, test=False):
    #Load the CONCERTO noise file

    nu_sens, NEFD_mJy_beam, Omega_beam_file = np.loadtxt(tex_file, unpack = True)
    Omega_beam = np.interp(nu_obs.to(u.GHz).value, nu_sens, Omega_beam_file*u.sr)
    #Rescale the sensitivity to the spectral resolution
    NEI_MJy_sr = (1e-9 * NEFD_mJy_beam / (Omega_beam) / dnu.value) * u.MJy
    #Needed to gen noisy maps
    NEFD_mJy_beam_obs =  np.interp(nu_obs.to(u.GHz).value, nu_sens, NEFD_mJy_beam) * u.mJy / dnu.value
    NEI_MJy_sr_obs   = np.interp(nu_obs.to(u.GHz).value, nu_sens, NEI_MJy_sr)

    if(test):
        input_spectra = Spectrum1D( flux=NEFD_mJy_beam*u.mJy, spectral_axis=nu_sens*u.GHz)
        '''
        n_slices = np.ceil(Dnu / dnu).value.astype(int)*3
        freqs = np.linspace(nu_obs-n_slices/2*dnu, nu_obs+n_slices/2*dnu, int(n_slices+1))
        '''
        freqs_1channel = np.asarray(((nu_obs-dnu).value,  nu_obs.value, (nu_obs+dnu).value))*u.GHz
        fluxc_resample = FluxConservingResampler()
        noise = fluxc_resample(input_spectra, freqs_1channel)  
        noise_for_slice = noise.flux[1]
        print(f'noise from Eq. 1 gives {np.round(NEFD_mJy_beam_obs,1)} and my test gives {np.round(noise_for_slice,1)}')
        #plt.plot(np.ones(len(nu_sens)), nu_sens, 'ob')
        plt.plot(freqs_1channel, np.ones(len(freqs_1channel)), 'og')
        plt.plot(noise.spectral_axis.bin_edges, np.ones(len(noise.spectral_axis.bin_edges)), 'ok')
        plt.plot(nu_sens, np.ones(len(nu_sens)), 'or')
    return NEI_MJy_sr_obs, NEFD_mJy_beam_obs.to(u.Jy), Omega_beam

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
    
    if(False):
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
    
    return cross_sensitivity_nsigma, w

def main_paper(simu, cat, simu_field_size, field_size_survey, line, rest_freq, z, dz, t_survey, npix, 
               dtype="_with_interlopers", n_slices=0.0, dnu_instru = 1.5*u.GHz):


    #Load the dicts for transition of reference 
    file = f"dict_dir/dict_LIMgal_{simu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{simu_field_size}deg2_{line}{dtype}.p"
    dict= pickle.load( open(file, 'rb'))
    file = f"dict_dir/dict_LIMgal_{simu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{simu_field_size}deg2_{line}.p"
    d = pickle.load( open(file, 'rb'))

    pkJG_interlopers = (np.abs(dict[f'pk_J-gal_{int(n_slices)}']) * u.sr).to(u.Jy)
    pkJ_interlopers =   np.abs(dict[f'pk_J_{int(n_slices)}'])


    #-----------------             
    #Without interlopers
    if(False): 
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
    
    zmin = z-(n_slices*dz)-dz/2 
    zmax = z+(n_slices*dz)+dz/2 
    freq_obs = rest_freq/(1+z)
    #dnu_simu = dz * freq_obs/(1+z) 
    Dnu = (zmax-zmin)*freq_obs/(1+z)
    NEI_MJy_sr, NEFD_Jy_beam, Omega_beam = sensitivity(dnu_instru*u.GHz, freq_obs) 
    
    Dc, delta_Dc, pk_3d_to_2d, k_3d_to_2d, full_volume_covered = cosmo_distance(z, Dnu, freq_obs, zmin, zmax) 
    #
    
    Vpix    = volume(Omega_beam, Dc, z, freq_obs, dnu_instru*u.GHz) 
    Vsurvey = field_size_survey.to(u.sr).value * full_volume_covered / (4*np.pi)
    t_obs = t_survey *  npix * Omega_beam.to(u.deg**2) / field_size_survey.to(u.deg**2)
    Pk_noise = (Vpix * NEI_MJy_sr**2/ t_obs).to(u.Jy**2/u.sr**2*u.Mpc**3) 
    
    pkJ =   np.abs(d[f'pk_J_{int(n_slices)}'])
    pkG =         (d[f'pk_gal_{int(n_slices)}']*u.sr**2).to(u.sr)
    pkJG = (np.abs(d[f'pk_J-gal_{int(n_slices)}']) * u.sr).to(u.Jy)
    pkJ_3d  = (pkJ   / pk_3d_to_2d).to(u.Jy**2/u.sr**2*u.Mpc**3)
    pkG_3d  = (pkG   / pk_3d_to_2d).to(u.Mpc**3)
    pkJG_3d = (pkJG  / pk_3d_to_2d).to(u.Jy/u.sr*u.Mpc**3)
    
    snJ = d['LIM_shot_list'][int(n_slices)]
    snG = d['gal_shot_list'][int(n_slices)]*u.sr
    snJG = (d['LIMgal_shot_list'][int(n_slices)]  * u.sr).to(u.Jy)
    snJ_3d  = (snJ   / pk_3d_to_2d).to(u.Jy**2/u.sr**2*u.Mpc**3)
    snG_3d  = (snG   / pk_3d_to_2d).to(u.Mpc**3)
    snJG_3d  = (snJG / pk_3d_to_2d).to(u.Jy/u.sr*u.Mpc**3)
    
    #res = d['res']
    k   = d['k']#.to(u.rad**-1)
    K   = k.to(u.rad**-1)/k_3d_to_2d #Mpc^-1
    delta_k = np.diff(d["kbin"]) ##!!
    delta_K = delta_k.to(u.rad**-1)/k_3d_to_2d #Mpc^-1
    Nmode = 2 * np.pi * (K**2 * delta_K) * Vsurvey/ (2 * np.pi)**3
    kmax = (1/np.sqrt(field_size_survey).to(u.rad)).to(1/u.arcmin)
    Klim = (1.8e-1/u.Mpc) #1.35
    klim = (Klim*k_3d_to_2d).to(1/u.arcmin)

    
    sigma_J  = sigma(pkJ_3d+Pk_noise,  Nmode) 
    sigma_G  = sigma(pkG_3d,           Nmode)
    sigma_JG = sigma(pkJ_3d+Pk_noise,  Nmode, pkG_3d, pkJG_3d)


    #w = np.where(k<klim)[0][-1]+1
    w = np.where(k>kmax)[0][0]
    SNR_J, SNR_K_J = SNR_fct(pkJ_3d, sigma_J, w) ##!!
    SNR_G, SNR_K_G = SNR_fct(pkG_3d, sigma_G, w)
    SNR_JG, SNR_K_JG = SNR_fct(pkJG_3d, sigma_JG, w)
    

    cross_sensitivity, w = plot_results(f"{simu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{simu_field_size}deg2_{line}", 
                simu, z, dz, line, t_survey, SNR_J, SNR_JG, Nmode,
                k_3d_to_2d, pk_3d_to_2d, 
                k, pkJ, snJ, sigma_J, pkG, snG, pkJG, snJG, sigma_JG, pkJG_interlopers,  
                klim, kmax, Pk_noise, pkJ_interlopers)


    
    return k, pkJ, snJ, pkG, snG, pkJG, snJG, k_3d_to_2d, pk_3d_to_2d, sigma_JG, SNR_K_JG, cross_sensitivity, w, kmax, SNR_JG, pkJG_interlopers
    
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

    for ip, (nslice, dz) in enumerate(zip(n_list,dz_list)): 
        if(ip !=1): continue
        
        fig, (power, SNR) = plt.subplots(2, 3, sharex=True, sharey = 'row', gridspec_kw={'height_ratios': [4,3]}, figsize=(8, 3.3), dpi = 200); lw=1.5
        patchs=[]
        for i, (line, linef, rest_freq, c) in enumerate(zip(line_list[2:5], line_list_fancy[2:5], rest_freq_list[2:5],('b', 'cyan', 'g'))):
            k, pkJ, snJ, pkG, snG, pkJG, snJG, k_3d_to_2d, pk_3d_to_2d, sigma_JG, SNR_JG, cross_sensitivity, w, klim, SNR_JGv,pkJG_interlopers = main_paper('pySIDES_from_uchuu', cat, 117, 
                                                           CONCERTO['Omega_field'], line, rest_freq, 
                                                           1.0, dz, CONCERTO['t_survey'], 
                                                           CONCERTO['npix'], n_slices=nslice, 
                                                           dnu_instru=CONCERTO['spectral_resolution'])
            #power[i].loglog(k.value, pkJG.value, linewidth=lw, color = 'k', ls=':')     
            power[i].loglog(k.value, pkJG_interlopers.value, linewidth=lw, color = 'gray')     

            power[i].axhline(snJG.value, color='k', linestyle='--',  linewidth=lw)
            power[i].axvline(klim.value, color='grey', linestyle='--',  linewidth=lw)
            power[i].loglog(k.value, pkJG.value-snJG.value, linewidth=lw, color = 'k')     
            power[i].fill_between(k.value, pkJG.value-(sigma_JG*pk_3d_to_2d).value,pkJG.value+(sigma_JG*pk_3d_to_2d).value, alpha=0.3, color=c)   
            power[i].loglog(k.value, cross_sensitivity*pk_3d_to_2d, linewidth=lw, color = 'orange')     
            SNR[i].loglog(k.value, SNR_JG, '--|', c=c,linewidth=lw, markersize=5)  
            SNR[i].legend(title=f'SNR tot.={np.round(SNR_JGv).astype(int)}', loc='upper left', fontsize=8, frameon=False)
            power[i].legend(title=f'{linef}@z=1.0', loc='upper right', fontsize=8, frameon=False)    
            if(i==1): SNR[i].set_xlabel('$\\rm k_{\\theta}$ [$\\rm arcmin^{-1}$]')
            if(i==0): SNR[i].set_ylabel('SNR')
            #power[i].legend( bbox_to_anchor=(1,1), fontsize=8, frameon=False)    

            patch = mpatches.Patch(color=c, label=linef); patchs.append(patch)

            power[i].tick_params(axis = "x", which='major', tickdir = "inout", color='k')
            power[i].tick_params(axis = "y", which='major', tickdir = "inout",  color='k')

            SNR[i].tick_params(axis = "y", which='major', tickdir = "inout", right = True, color='k')
            SNR[i].tick_params(axis = "y", which='minor', left=False, color='k')


            def g(k_3d_to_2d, x): return (x*u.arcmin**-1).to(u.rad**-1).value / k_3d_to_2d.value
            G = partial( g, k_3d_to_2d )
            def f(pk_3d_to_2d, x): return x / pk_3d_to_2d.value
            F = partial( f, pk_3d_to_2d )
            secax =power[i].secondary_xaxis("top", functions=(G,G))
            if(i==1): secax.set_xlabel('k [$\\rm Mpc^{-1}$]')
            secax =power[i].secondary_yaxis("right", functions=(F,F))
            if(i==2): secax.set_ylabel('$\\rm P_{\\times}$(k) [$\\rm Jy.sr^{-1}.Mpc^3]$')
            if(i==0): power[i].set_ylabel('$\\rm P_{\\times}(k_{\\theta})$ [$\\rm Jy]$')

        plt.rcParams.update({'font.size': 10})
        #plt.rcParams.update({'legend.frameon':False})
        #fig.tight_layout()
        fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust rect to make space on the right
        fig.subplots_adjust(hspace=0, wspace=0)


        patch = mlines.Line2D([], [], color='k', linestyle="solid", label='Clustering' ); patchs.append(patch);
        patch = mlines.Line2D([], [], color='k', linestyle="--",    label='Shot noise' ); patchs.append(patch);
        patch = mlines.Line2D([], [], color='grey',                 label='$\\rm P^{tot}_{\\times}(k)$ \n+interlopers' ); patchs.append(patch);
        patch = mlines.Line2D([], [], color='orange',                    label='CONCERTO \n 3$\\rm \\sigma$ limit \n' ); patchs.append(patch);
        patch = mpatches.Patch(color='gray', label='r.m.s \n on $P^{tot}_{\\times}$'); patchs.append(patch)

        patch = mlines.Line2D([], [], color='grey', linestyle="--",    label='$\\rm k^{CONCERTO}_{min}$' ); patchs.append(patch);

        fig.legend(handles=patchs, bbox_to_anchor=(0.99, 0.8), fontsize=8, frameon=False)
        fig.savefig(f'nslice{nslice}_z1_dz{dz}.png', transparent=True)


        fig, (power, SNR) = plt.subplots(2, 3, sharex=True, sharey = 'row', gridspec_kw={'height_ratios': [4,3]}, figsize=(8, 3.3), dpi = 200); lw=1.5
        patchs=[]
        for i,(line, linef, rest_fre, c) in enumerate(zip(line_list[3:6],line_list_fancy[3:6],rest_freq_list[3:6], ('cyan', 'g', 'purple'))):        
            k, pkJ, snJ, pkG, snG, pkJG, snJG, k_3d_to_2d, pk_3d_to_2d, sigma_JG, SNR_JG, cross_sensitivity, w, klim, SNR_JGv, pkJG_interlopers = main_paper('pySIDES_from_uchuu', cat, 117, 
                                                           CONCERTO['Omega_field'], line, rest_freq, 
                                                           1.5, dz, CONCERTO['t_survey'], CONCERTO['npix'], 
                                                           n_slices=nslice, dnu_instru=CONCERTO['spectral_resolution'])
        
            #power[i].loglog(k.value, pkJG.value, linewidth=lw, color = 'k', ls=':')     
            power[i].loglog(k.value, pkJG_interlopers.value, linewidth=lw, color = 'gray')     

            power[i].axhline(snJG.value, color='k', linestyle='--',  linewidth=lw)
            power[i].axvline(klim.value, color='grey', linestyle='--',  linewidth=lw)
            power[i].loglog(k.value, pkJG.value-snJG.value, linewidth=lw, color = 'k')     
            power[i].fill_between(k.value, pkJG.value-(sigma_JG*pk_3d_to_2d).value,pkJG.value+(sigma_JG*pk_3d_to_2d).value, alpha=0.3, color=c)   
            power[i].loglog(k.value, cross_sensitivity*pk_3d_to_2d, linewidth=lw, color = 'orange')     
            SNR[i].loglog(k.value, SNR_JG, '--|', c=c,linewidth=lw, markersize=5)  
            SNR[i].legend(title=f'SNR tot.={np.round(SNR_JGv).astype(int)}', loc='upper left', fontsize=8, frameon=False)
            power[i].legend(title=f'{linef}@z=1.5', loc='upper right', fontsize=8, frameon=False)    
            if(i==1): SNR[i].set_xlabel('$\\rm k_{\\theta}$ [$\\rm arcmin^{-1}$]')
            if(i==0): SNR[i].set_ylabel('SNR')
            #power[i].legend( bbox_to_anchor=(1,1), fontsize=8, frameon=False)    

            patch = mpatches.Patch(color=c, label=linef); patchs.append(patch)

            power[i].tick_params(axis = "x", which='major', tickdir = "inout", color='k')
            power[i].tick_params(axis = "y", which='major', tickdir = "inout",  color='k')
            power[i].tick_params(axis = "y", which='minor',left =False)
            SNR[i].tick_params(axis = "y", which='major', tickdir = "inout", right = True, color='k')
            SNR[i].tick_params(axis = "y", which='minor', left=False)

            def g(k_3d_to_2d, x): return (x*u.arcmin**-1).to(u.rad**-1).value / k_3d_to_2d.value
            G = partial( g, k_3d_to_2d )
            def f(pk_3d_to_2d, x): return x / pk_3d_to_2d.value
            F = partial( f, pk_3d_to_2d )
            secax =power[i].secondary_xaxis("top", functions=(G,G))
            if(i==1): secax.set_xlabel('k [$\\rm Mpc^{-1}$]')
            secax =power[i].secondary_yaxis("right", functions=(F,F))
            secax.tick_params(axis = "y", which='minor',right=False)

            if(i==2): secax.set_ylabel('$\\rm P_{\\times}$(k) [$\\rm Jy.sr^{-1}.Mpc^3]$')
            if(i==0): power[i].set_ylabel('$\\rm P_{\\times}(k_{\\theta})$ [$\\rm Jy]$')


        plt.rcParams.update({'font.size': 10})
        #plt.rcParams.update({'legend.frameon':False})
        #fig.tight_layout()
        fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust rect to make space on the right
        fig.subplots_adjust(hspace=0, wspace=0)


        patch = mlines.Line2D([], [], color='k', linestyle="solid", label='Clustering' ); patchs.append(patch);
        patch = mlines.Line2D([], [], color='k', linestyle="--",    label='Shot noise' ); patchs.append(patch);
        patch = mlines.Line2D([], [], color='grey',                 label='$\\rm P^{tot}_{\\times}(k)$ \n+interlopers' ); patchs.append(patch);
        patch = mlines.Line2D([], [], color='orange',                    label='CONCERTO \n 3$\\rm \\sigma$ limit \n' ); patchs.append(patch);
        patch = mpatches.Patch(color='gray', label='r.m.s \n on $P^{tot}_{\\times}$'); patchs.append(patch)

        patch = mlines.Line2D([], [], color='grey', linestyle="--",    label='$\\rm k^{CONCERTO}_{min}$' ); patchs.append(patch);

        fig.legend(handles=patchs, bbox_to_anchor=(0.99, 0.8), fontsize=8, frameon=False)
        fig.savefig(f'nslice{nslice}_z1.5_dz{dz}.png', transparent=True)

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