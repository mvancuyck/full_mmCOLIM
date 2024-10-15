from fcts import * 
from gen_all_sizes_cubes_and_cat import * 
from functools import partial
from multiprocessing import Pool, cpu_count
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
import argparse
from pysides.load_params import *

line_list_fancy = ["CO({}-{})".format(J_up, J_up - 1) for J_up in range(1, 9)]
#line_list.append('CI10'); line_list.append('CI21'); 
line_list_fancy.append('[CII]')

def fit_b_gal(angular_k,pk_matter_2d, k, pk_without_sn, sigma_pk ):
    
    def model_auto_gal_to_fit(angular_k,pk_matter_2d, k, b):
        f = interpolate.interp1d(angular_k, pk_matter_2d,  kind='linear')
        pk_matter_nonlin_rebin = f(k)
        return  pk_matter_nonlin_rebin * b**2
    
    F = partial( model_auto_gal_to_fit, angular_k.value,  pk_matter_2d)
    popt_J, pcov_J = curve_fit(F, k, pk_without_sn, sigma = sigma_pk, p0 = (1), bounds = (0,(6)) )#maxfev=20000,
    b = popt_J[0]; delta_b = np.sqrt(pcov_J[0][0])
    return b, delta_b, 0, 0

def fit_b_eff_auto(angular_k,pk_matter_2d, k, pk_without_sn, sigma_pk, I, delta_I, embed_bool = False ):

    if(embed_bool): embed()
   
    def model_auto_line_to_fit(angular_k,pk_matter_2d, I, k, b_co):
        f = interpolate.interp1d(angular_k, pk_matter_2d,  kind='linear')
        pk_matter_nonlin_rebin = f(k)
        return  pk_matter_nonlin_rebin * (I*b_co)**2

    F = partial( model_auto_line_to_fit, angular_k.value,  pk_matter_2d, I)
    if((sigma_pk==0).all()): sigma_pk = np.ones(1e-2)
    popt_J, pcov_J = curve_fit(F, k, pk_without_sn, sigma = sigma_pk, p0 = (1), bounds = (0,10) )#maxfev=20000,
    b = popt_J[0]
    delta_b = np.sqrt(pcov_J[0][0])
    return b, delta_b

def fit_b_gal_nonlin(angular_k,pk_matter_2d, k, pk_without_sn, sigma_pk, embed_bool = False ):
    if(embed_bool): embed()

    def model_auto_gal_to_fit(angular_k,pk_matter_2d, k, b, B):
        f = interpolate.interp1d(angular_k, pk_matter_2d,  kind='linear')
        pk_matter_nonlin_rebin = f(k)
        return  pk_matter_nonlin_rebin * (b**2)+B #pk_matter_nonlin_rebin * (b**2+B)
    
    F = partial( model_auto_gal_to_fit, angular_k.value,  pk_matter_2d)
    popt_J, pcov_J = curve_fit(F, k, pk_without_sn, sigma = sigma_pk, p0 = (1,0), bounds = ((0,0),(10,10)) )#maxfev=20000,
    b = popt_J[0]; delta_b = np.sqrt(pcov_J[0][0])
    B = popt_J[1]; delta_B = np.sqrt(pcov_J[1][1])

    return b, delta_b, B, delta_B

def fit_b_eff_nonlin(angular_k, pk_matter_2d, k, pk_without_sn, sigma_pk, I, delta_I, embed_bool = False ):

    if(embed_bool): embed()
   
    def model_auto_line_to_fit(angular_k, pk_matter_2d, I, k, b_co, B):
        f = interpolate.interp1d(angular_k, pk_matter_2d,  kind='linear')
        pk_matter_nonlin_rebin = f(k)
        return pk_matter_nonlin_rebin * ((I*b_co)**2 + B) #pk_matter_nonlin_rebin * ((I*b_co)**2 + B)

    F = partial( model_auto_line_to_fit, angular_k.value, pk_matter_2d, I)
    if((sigma_pk==0).all()): sigma_pk = np.ones(1e-2)
    popt_J, pcov_J = curve_fit(F, k, pk_without_sn, sigma = sigma_pk, p0 = (1,0), bounds = ((0,0),(10,(I*10)**2) ) )
    b = popt_J[0]; delta_b = np.sqrt(pcov_J[0][0])
    B = popt_J[1]; delta_B = np.sqrt(pcov_J[1][1])

    return b, delta_b, B, delta_B

def bI_from_autospec_and_cat(n_slices, field_size, ksimu, z_list, dz_list,
                       allpoints = False, recompute=False):
    
    list_I =  np.zeros((12,len(z_list), len(dz_list), len(line_list[:8])))
    list_Gal= np.zeros((12,len(z_list), len(dz_list), len(line_list[:8]),27))
    list_J =  np.zeros((12,len(z_list), len(dz_list), len(line_list[:8]),27))
    list_K =  np.zeros((12,len(z_list), len(dz_list), len(line_list[:8]),27))
    list_k =  np.zeros((12,len(z_list), len(dz_list), len(line_list[:8]),27))

    for nfield in range(12):
        for zi, z in enumerate(z_list):
            for dzi, dz in enumerate(dz_list):
                for j, (J, rest_freq) in enumerate(zip(line_list[:8], rest_freq_list[:8])):

                    #-----------------                
                    #Without interlopers
                    file = f"dict_dir/dict_LIMgal_pySIDES_from_uchuu_ntile_{nfield}_z{z}_dz{dz}_{n_slices}slices_{field_size}deg2_{J}.p"
                    dict = pickle.load( open(file, 'rb'))
                    #-----------------
                    K = ((dict["k"].to(u.rad**-1) * 2 * np.pi  /  dict["Dc"]).value)[:27]
                    list_K[nfield, zi, dzi, j,:K.shape[0]] = K
                    list_k[nfield, zi, dzi, j,:K.shape[0]] = (dict["k"].to(u.arcmin**-1).value)[:27]

                    list_I[nfield, zi, dzi, j] = dict["I"][0].value
                    G = (dict["pk_gal"][0].value-dict["gal_shot"][0])[:27]
                    list_Gal[nfield, zi, dzi, j,:] = G
                    JJ = (dict["pk_J"][0].value - dict["LIM_shot"][0].value )[:27]                            
                    list_J[nfield, zi, dzi, j,:] = JJ

    b_list_nonpoiss = np.zeros((12,len(z_list), len(dz_list), len(line_list[:8]), 4, 2))
    b_list = np.zeros((12,len(z_list), len(dz_list), len(line_list[:8]), 2, 2))

    for nfield in range(12):
        for zi, z in enumerate(z_list):
            for dzi, dz in enumerate(dz_list):
                for j, (J, rest_freq) in enumerate(zip(line_list[:8], rest_freq_list[:8])):

                    #Without interlopers
                    file = f"dict_dir/dict_LIMgal_pySIDES_from_uchuu_ntile_{nfield}_z{z}_dz{dz}_{n_slices}slices_{field_size}deg2_{J}.p"
                    dict = pickle.load( open(file, 'rb'))

                    K = list_K[nfield, zi, dzi, j, :]
                    w = np.where( K <= ksimu)
                    k_a = w[0][0]; k_e =  w[0][-1]
                    if(k_e <= 1):  k_e = 2
                    if(allpoints): born = [0,-1]
                    else: born = [k_a,k_e+1]
                    #[born[0]:born[1]]
                    K = list_K[nfield, zi, dzi, j, :][born[0]:born[1]]
                    k = list_k[nfield, zi, dzi, j, :][born[0]:born[1]]

                    I = list_I[nfield, zi, dzi, j]
                    Gal = list_Gal[nfield, zi, dzi, j, :][born[0]:born[1]]
                    JJ = list_J[nfield, zi, dzi, j, :][born[0]:born[1]]
                    sigma_Gal  = np.std(list_Gal[:, zi, dzi, j, :], axis=0)[born[0]:born[1]]
                    sigma_JJ   = np.std(list_J[:, zi, dzi, j, :], axis=0)[born[0]:born[1]]
                    sigma_I   = np.std(list_I[:, zi, dzi, j])
                    
                    bgal,dbgal, Bgal, dBgal = fit_b_gal(dict["k_angular"], dict["pk_matter_2d"], k, Gal, sigma_Gal) #Can change
                    bX, dbX, B, dB          = fit_b_eff_nonlin(dict["k_angular"], dict["pk_matter_2d"], k, JJ,  sigma_JJ, 
                                               I, sigma_I )
                    
                    b_list_nonpoiss[nfield, zi, dzi, j,0,0] = bX
                    b_list_nonpoiss[nfield, zi, dzi, j,0,1] = dbX
                    b_list_nonpoiss[nfield, zi, dzi, j,1,0] = bgal
                    b_list_nonpoiss[nfield, zi, dzi, j,1,1] = dbgal
                    b_list_nonpoiss[nfield, zi, dzi, j,2,0] = B
                    b_list_nonpoiss[nfield, zi, dzi, j,2,1] = dB
                    b_list_nonpoiss[nfield, zi, dzi, j,3,0] = Bgal
                    b_list_nonpoiss[nfield, zi, dzi, j,3,1] = dBgal
                    
                    bgal_lin,dbgal_lin, _, _  = fit_b_gal(     dict["k_angular"], dict["pk_matter_2d"], k, Gal, sigma_Gal) #Cannot change
                    bX_lin, dbX_lin     = fit_b_eff_auto(dict["k_angular"], dict["pk_matter_2d"], k, JJ,  sigma_JJ, 
                                               I, sigma_I )
                    
                    b_list[nfield, zi, dzi, j,0,0] = bX_lin
                    b_list[nfield, zi, dzi, j,0,1] = dbX_lin
                    b_list[nfield, zi, dzi, j,1,0] = bgal_lin
                    b_list[nfield, zi, dzi, j,1,1] = dbgal_lin
                    
                    #---------------------
                    if(True and nfield==0 and J=='CO32'):

                        k_2d_to_3d  = dict['Dc']*2*np.pi 
                        pk_2d_to_3d = dict['Dc']**2*dict['delta_Dc']    
                        def g(k_2d_to_3d, x): return (x*u.arcmin**-1).to(u.rad**-1).value / k_2d_to_3d.value
                        G = partial( g, k_2d_to_3d )
                        def h(pk_2d_to_3d, x): return x * pk_2d_to_3d.value
                        F = partial( h, pk_2d_to_3d )

                        BS=7; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); lw=1; mk=5; elw=1
                        fig, (axj, axg) = plt.subplots(2, 1, sharex=True, sharey = 'row', 
                                                       gridspec_kw={'height_ratios': [1,1]}, figsize=(3,5), dpi = 200)
                        #axj.set_title(f'n={nfield}, z={z}, dz={dz}, {J}')
                        #--- data ---
                        axj.set_ylim(2e-5, 1)
                        axj.set_xlim(5e-3, 5e-1)
                        axj.loglog( dict["k"], dict["pk_J"][0].value,
                                   c='darkgray', ls="--", label = 'non-param. $\\rm P^{tot}$(k)' ,lw=lw, zorder=1)
                        axj.errorbar( dict["k"], dict["pk_J"][0].value, yerr=np.std(list_J[:, zi, dzi, j, :], axis=0),
                                   c='darkgray', ls="--", lw=lw)
                        
                        axj.axhline(dict["LIM_shot"][0].value, linestyle=':', 
                                   c='darkgray', ls=':',  label = 'shot noise',                   lw=lw, zorder=2 )
                        axj.loglog( dict["k"], dict["pk_J"][0].value - dict["LIM_shot"][0].value, 
                                   c='darkgray', lw=lw,   label = 'non param. $\\rm P^{clust}$(k)',      zorder=3 )
                        axj.loglog( k, JJ, 'or',markersize=mk, label='points used in fit', zorder=5)
                        axg.errorbar( dict["k"], dict["pk_gal"][0].value, yerr=np.std(list_Gal[:, zi, dzi, j, :], axis=0),  c='darkgray', ls="--",)
                        axg.axhline(dict["gal_shot"][0], linestyle=':', c='darkgray', ls=':', lw=lw, )
                        axg.loglog( dict["k"], dict["pk_gal"][0].value - dict["gal_shot"][0], c='darkgray', lw=lw,)
                        axg.loglog( k, Gal, 'or',markersize=mk)
                        #---  ---

                        f = interp1d( dict['k_angular'],  dict['pk_matter_2d'] )
                        axj.loglog( k, f(k) * B , c='k', ) #+ dict["LIM_shot"][0].value
                                   
                        axj.loglog( k, f(k) * ( bX_lin*I)**2, '--xb',
                                   label ='$\\rm (B_{\\nu} \\times b^{\\rm Poisson}_\\mathrm{eff})^2 P^{\\rm 2D}_{\\rm matter}$ \n$ b^{\\rm Poisson}_\\mathrm{eff}$='+f'{np.round(bX_lin,2)}'+'$\\pm$'+f'{np.round(dbX_lin,2)}', lw=lw,markersize=mk, zorder=6)
                        axj.loglog( k, f(k) * ((bX*I)**2+B), '.',c='orange',
                                   label ='$\\rm [(B_{\\nu} \\times b_\\mathrm{eff})^2 + C_\\mathrm{eff} ] P^{\\rm 2D}_{\\rm matter}$ \n$ b^{\\rm non-Poisson}_\\mathrm{eff}$='+f'{np.round(bX,2)}'+'$\\pm$'+f'{np.round(dbX,0)}', lw=lw,markersize=mk, zorder=6)
                        axg.loglog( k, f(k)*bgal_lin**2, '--xg', lw=lw,markersize=mk, 
                                   label ='$ (b^{\\rm Poisson}_\\mathrm{gal})^2 P^{\\rm 2D}_{\\rm matter}$ \n$ b^{\\rm Poisson}_\\mathrm{gal}$='+f'{np.round(bgal_lin,2)}'+'$\\pm$'+f'{np.round(dbgal_lin,2)}')                        
                        axg.loglog( k, f(k)*(bgal**2+Bgal), '.', lw=lw,markersize=mk, c='orange',
                                   label ='$ [(b_\\mathrm{gal})^2+C] P^{\\rm 2D}_{\\rm matter}$ \n$ b^{\\rm non-Poisson}_\\mathrm{gal}$='+f'{np.round(bgal,2)}'+'$\\pm$'+f'{np.round(dbgal,0)}')                        
                        axg.loglog( k, f(k) * Bgal , c='k',label='non-Poisson component of shot noise') #+ dict["gal_shot"][0]

                        secax = axj.secondary_xaxis("top", functions=(G,G))
                        secax.set_xlabel('k [$\\rm Mpc^{-1}$]')

                        secax = axj.secondary_yaxis("right", functions=(F,F))
                        secaxg = axg.secondary_yaxis("right", functions=(F,F))

                        axj.legend(fontsize=BS-2,frameon=False, loc = 'lower left')
                        axg.legend(fontsize=BS-2,loc = 'lower left', frameon=False)
                        axj.set_ylabel("$\\rm P_{"+'{}'.format(J)+"}(k_{\\theta})$ " +"[$\\rm Jy^2/sr$]"); #plt.ylim(5e-6,2e-3)
                        axg.set_ylabel("$\\rm P_{gal}$($\\rm k_{\\theta}$) [sr]")
                        axg.set_xlabel("$\\rm k_{\\theta}$ [$\\rm arcmin^{-1}$]")#, z={z}, dz={dz}")
                        secax.set_ylabel("$\\rm P_{"+'{}'.format(J)+"}(k)$ "+' [$\\rm Jy^2/sr^2.Mpc^3]$')
                        secaxg.set_ylabel('$\\rm P_{gal}$(k) [$\\rm Mpc^3]$')
                        fig.tight_layout()
                        fig.subplots_adjust(hspace=.0)
                        for extension in ("png",): plt.savefig(f"example_{J}_z{z}_dz{dz}_n{nfield}_nslice{n_slices}_fit_nonpoisson.{extension}")
                        plt.close()
    
    return b_list, list_I, b_list_nonpoiss

def bI_crosspec(n_slices, field_size, ksimu, z_list, dz_list, b_list,
                       allpoints = False, recompute=False, dtype=''):
    
    bI_mes = np.zeros((12,len(z_list), len(z_list), len(line_list[:8])))
    list_k = np.zeros((12,len(z_list), len(z_list), len(line_list[:8]),27))
    bI_mesCI = np.zeros((12,len(z_list), len(z_list),))
    bI_mesCO76 = np.zeros((12,len(z_list), len(z_list),))

    for nfield in range(12):
        for zi, z in enumerate(z_list):
            for dzi, dz in enumerate(dz_list):
                for j, (J, rest_freq) in enumerate(zip(line_list[:8], rest_freq_list[:8])):

                    #-----------------                
                    #Without interlopers
                    file = f"dict_dir/dict_LIMgal_pySIDES_from_uchuu_ntile_{nfield}_z{z}_dz{dz}_{n_slices}slices_{field_size}deg2_{J}{dtype}.p"
                    dict = pickle.load( open(file, 'rb'))
                    file = f"dict_dir/dict_LIMgal_pySIDES_from_uchuu_ntile_{nfield}_z{z}_dz{dz}_{n_slices}slices_{field_size}deg2_{J}.p"
                    d = pickle.load( open(file, 'rb'))
                    #-----------------
                    K = ((dict["k"].to(u.rad**-1) * 2 * np.pi  /  d["Dc"]).value)[:27]
                    w = np.where( K <= ksimu)
                    k_a = w[0][0]; k_e =  w[0][-1]
                    if(k_e <= 1):  k_e = 2
                    if(allpoints): born = [0,-1]
                    else: born = [k_a,k_e+1]
                    #[born[0]:born[1]]
                    k = (dict["k"].to(u.arcmin**-1).value)[born[0]:born[1]]
                    f = interpolate.interp1d(  d["k_angular"], d["pk_matter_2d"],  kind='linear')
                    p2d = f(k)
                    Jgal = dict["pk_J-gal"][0][born[0]:born[1]].value - d["LIMgal_shot"][0].value
                    w = np.where(Jgal>0)
                    bI_mes[nfield, zi, dzi, j]  =( Jgal / b_list[nfield, zi, dzi, j, 1,0] / p2d )[w].mean()

                    if(False and nfield==0 and J=='CO87'):
                        plt.figure()
                        plt.loglog(dict["k"].to(u.arcmin**-1).value, dict["pk_J-gal"][0] )
                        plt.loglog(dict["k"].to(u.arcmin**-1).value, dict["pk_J-gal"][0].value- d["LIMgal_shot"][0].value )
                        plt.axhline(d["LIMgal_shot"][0].value, linestyle=':', c='darkgray', ls=':' )
                        plt.loglog(k, dict["pk_J-gal"][0][born[0]:born[1]].value, 'or')
                        plt.loglog(k[w], p2d[w], ':og')
                        plt.loglog(k[w], (Jgal / b_list[nfield, zi, dzi, j, 1,0] / p2d)[w], '--xb')
                        plt.xlabel(f'k arcmin-1, b={b_list[nfield, zi, dzi, j, 1,0]}')
                        plt.savefig(f'bI_n{nfield}_z{z}_dz{dz}_{J}{dtype}.png')
                        plt.close()

                    if(J=='CO76'):
                        Jgal = d["pk_J-gal"][0][born[0]:born[1]].value - d["LIMgal_shot"][0].value
                        bI_mesCO76[nfield, zi, dzi]  =( Jgal / b_list[nfield, zi, dzi, j, 1,0] / p2d )[w].mean()
                        
                        file = f"dict_dir/dict_LIMgal_pySIDES_from_uchuu_ntile_{nfield}_z{z}_dz{dz}_{n_slices}slices_{field_size}deg2_CI21.p"
                        d = pickle.load( open(file, 'rb'))
                        Jgal = d["pk_J-gal"][0][born[0]:born[1]].value - d["LIMgal_shot"][0].value
                        bI_mesCI[nfield, zi, dzi]  =( Jgal / b_list[nfield, zi, dzi, j, 1,0] / p2d )[w].mean()
                        
                    
    return bI_mes, bI_mesCO76, bI_mesCI

def bI_from_tinker(n_slices, field_size, ksimu, z_list, dz_list,
                       allpoints = False, recompute=False):
    
    list_bI =  np.zeros((12,len(z_list), len(dz_list), len(line_list[:8])))
    list_b =  np.zeros((12,len(z_list), len(dz_list), len(line_list[:8])))

    for nfield in range(12):
        for zi, z in enumerate(z_list):
            for dzi, dz in enumerate(dz_list):
                for j, (J, rest_freq) in enumerate(zip(line_list[:8], rest_freq_list[:8])):

                    #-----------------                
                    #Without interlopers
                    file = f"dict_dir/dict_LIMgal_pySIDES_from_uchuu_ntile_{nfield}_z{z}_dz{dz}_{n_slices}slices_{field_size}deg2_{J}.p"
                    dict = pickle.load( open(file, 'rb'))
                    #-----------------
                    list_bI[nfield, zi, dzi, j] = dict["I"][0].value * dict['beff_t10'][0]
                    list_b[nfield, zi, dzi, j] =  dict['beff_t10'][0]

    return list_bI , list_b

def b_I(n_slices, z_list, dz_list, ref = 'CO32',  dtype='all_lines', recompute=False):

    colors_co = ('orange', 'r', 'b', 'cyan', 'g', 'purple', 'magenta', 'grey',)
    b_list, I_list, b_list_nonpoiss = bI_from_autospec_and_cat(n_slices,9,0.15, z_list, dz_list )
    bI_tinker, b_tinker = bI_from_tinker(n_slices,9,0.15, z_list, dz_list)
    bI_mes, bI_mesCO76, bI_mesCI = bI_crosspec(n_slices,9,0.15, z_list, dz_list, b_list,dtype = '_with_interlopers')
    bI_mes_nonpoiss, bI_mesCO76_nonpoiss, bI_mesCI_nonpoiss = bI_crosspec(n_slices,9,0.15, z_list, dz_list, b_list_nonpoiss,dtype = '_with_interlopers')

    idz=0

    #----------------------------------------------
    patchs=[]
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(5,5), dpi=200); 
    mks=6; dr = 0.04; dy = 0.0; text_y_offset=0.2; lw=1; width = [1,1];  height = [1,1,1,1,1]
    drelative = (-4,-3,-2,-1,0,1,2,3)
    gs = gridspec.GridSpec(ncols=2, nrows=5, height_ratios= height, width_ratios=width)
    axr = plt.subplot(gs[-2]); patchs = []
    axr.set_xlabel(r"redshift"); axr.set_ylabel("relative \n difference"); axr.set_ylim(-0.3, 1.2);
    for j, (line, rest_freq) in enumerate(zip(line_list_fancy[:8], rest_freq_list[:8])):
        c=colors_co[j]; 
        #---------------------------
        ax = plt.subplot(gs[j]); 
        #---------------------------
        ax.errorbar(    z_list, np.mean(bI_mes[:,:,idz,j], axis=0),  linestyle='solid', color=c, ecolor=c, lw=lw) 
        ax.fill_between(z_list, np.mean(bI_mes[:,:,idz,j], axis=0)-np.std(bI_mes[:,:,idz,j], axis=0), np.mean(bI_mes[:,:,idz,j], axis=0)+np.std(bI_mes[:,:,idz,j], axis=0), alpha = 0.2, color=c)
        #---
        ax.errorbar(    z_list, np.mean(bI_mes_nonpoiss[:,:,idz,j], axis=0),  linestyle='solid', color='k', ecolor='k', lw=lw) 
        ax.fill_between(z_list, np.mean(bI_mes_nonpoiss[:,:,idz,j], axis=0)-np.std(bI_mes_nonpoiss[:,:,idz,j], axis=0), np.mean(bI_mes_nonpoiss[:,:,idz,j], axis=0)+np.std(bI_mes_nonpoiss[:,:,idz,j], axis=0), alpha = 0.2, color='gray')
        #---
        ax.errorbar( z_list, np.mean(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0), yerr=np.std(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0),
                     linestyle=":", color=c, ecolor=c, marker='*',markersize = mks,lw=lw )
        #---
        ax.errorbar( z_list, np.mean(b_list_nonpoiss[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0), yerr=np.std(b_list_nonpoiss[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0),
                     linestyle=":", color=c, ecolor=c, marker='>',markersize = mks,lw=lw,mfc=c, mec='k' )
        #---
        #ax.errorbar( z_list, np.mean(bI_tinker[:, :, idz, j], axis=0), yerr=np.std(bI_tinker[:, :, idz, j], axis=0),
        #             linestyle="dashdot", color='k', ecolor='k', marker="p",markersize = mks,lw=lw,  mfc='none', mec='k')
        axr.errorbar( np.asarray(z_list)+drelative[j]*dr, 
                     (np.mean(b_list_nonpoiss[:, :, idz, j,0,0]*I_list[:,:,idz,j]/(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j]), axis=0))-1+dy, 
                     yerr = (np.std(b_list_nonpoiss[:, :, idz, j,0,0]*I_list[:,:,idz,j]/(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j]), axis=0)), 
                     linestyle="None", color=c,  ecolor=c, marker='*', markersize = mks, lw=lw )
    #---
        axr.set_ylim(-0.15,0.15)
        #---------------------------
        ax.set_ylabel(r"$b_{\rm eff} \times$"+"$\\rm B^{"+'{}'.format(line)+"}_{\\nu}$ ")
        ax.set_xlim(0.4, np.max(z_list)+0.1)
        ax.set_yscale("log")
        if(j!=7): ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=False)
        else:
            ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=True)
        if(j == 7): ax.set_xlabel(r"redshift")
    fig.tight_layout(); fig.subplots_adjust(hspace=.0)
    patch = mlines.Line2D([], [], color='k', linestyle=":", marker="*",  label='catalogue mean brightness \ntimes effective bias'); patchs.append(patch)
    patch = mlines.Line2D([], [], color='k', linestyle="dashdot", marker="p", mfc='none', label=r'Tinker et al. 2010'); patchs.append(patch)
    for extension in ("png", "pdf"): plt.savefig(f"bI_CO_nslice{nslice}_dz{dz_list[0]}_test_non_poissonian_shot_noise.{extension}", transparent=True)
    #----------------------------------------------

    #----------------------------------------------
    BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig,axs = plt.subplots(2,2,figsize=(9,5), dpi=200); 
    axb, axbgal, axB, axBgal = axs[0,0], axs[0,1], axs[1,0], axs[1,1]
    axb.set_xlabel('redshift'); axbgal.set_xlabel('redshift'); axB.set_xlabel('redshift'); axBgal.set_xlabel('redshift')
    axb.set_ylabel('effective CO bias $\\rm b_{eff}(z)$'); axbgal.set_ylabel('galaxy bias b(z)'); 
    axB.set_ylabel('Non-poiss. param $\\rm C_{eff}$ in \n CO auto power'); axBgal.set_ylabel('Non-poiss. param C in \n galaxies auto power')
    for j, (line, rest_freq) in enumerate(zip(line_list_fancy[:8], rest_freq_list[:8])):
        c=colors_co[j]; 
        #---
        axb.errorbar( z_list, np.mean(b_list[:, :, idz, j,0,0], axis=0), yerr=np.std(b_list[:, :, idz, j,0,0], axis=0),
                     linestyle=":", color=c, ecolor=c, marker='*',markersize = mks,lw=lw, mec='k' )
        axb.errorbar( z_list, np.mean(b_tinker[:, :, idz, j], axis=0), yerr=np.std(b_tinker[:, :, idz, j], axis=0),
                     linestyle="dashdot", color='k', ecolor='k', marker="p",markersize = mks,lw=lw,  mfc='none', mec='k')
        axb.errorbar( z_list, np.mean(b_list_nonpoiss[:, :, idz, j,0,0], axis=0), yerr=np.std(b_list_nonpoiss[:, :, idz, j,0,0], axis=0),
                     linestyle=":", color=c, ecolor=c, marker='>',markersize = mks,lw=lw )
        #---
        axB.errorbar( z_list, np.mean(b_list_nonpoiss[:, :, idz, j,2,0], axis=0), yerr=np.std(b_list_nonpoiss[:, :, idz, j,2,0], axis=0),
                     linestyle=":", color=c, ecolor=c, marker='>',markersize = mks,lw=lw )
        #---
    axbgal.errorbar( z_list, np.mean(b_list[:, :, idz, :,1,0], axis=(0,2)), yerr=np.std(b_list[:, :, idz, :,1,0], axis=(0,2)),
                     linestyle=":", color='k', ecolor='k', marker='*',markersize = mks,lw=lw )
    axbgal.errorbar( z_list, np.mean(b_list_nonpoiss[:, :, idz, :,1,0], axis=(0,2)), yerr=np.std(b_list_nonpoiss[:, :, idz, :,1,0], axis=(0,2)),
                     linestyle=":", color=c, ecolor=c, marker='>',markersize = mks,lw=lw )
    axBgal.errorbar( z_list, np.mean(b_list_nonpoiss[:, :, idz, :,3,0], axis=(0,2)), yerr=np.std(b_list_nonpoiss[:, :, idz, :,3,0], axis=(0,2)),
                     linestyle=":", color=c, ecolor=c, marker='>',markersize = mks,lw=lw )
    patchs=[]
    patch = mlines.Line2D([], [], color='k', linestyle=":", marker="*",  label='fit with Poissonian shot noise'); patchs.append(patch)
    patch = mlines.Line2D([], [], color='k', linestyle=":", marker=">",  label='fit with non-Poissonian shot noise'); patchs.append(patch)
    patch = mlines.Line2D([], [], color='k', linestyle="dashdot", marker="p",  label='Tinker et al. 2010'); patchs.append(patch)
    for j, (line, rest_freq) in enumerate(zip(line_list_fancy[:8], rest_freq_list[:8])):
        patch = mlines.Line2D([], [], linestyle="None", color=colors_co[j], marker="o",  label=line); patchs.append(patch)
    axbgal.legend(handles = patchs,  bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    fig.tight_layout(); 
    #----------------------------------------------

    for extension in ("png", "pdf"): plt.savefig(f"bI_CO_nslice{nslice}_dz{dz_list[0]}_fitfig_test_non_poissonian_shot_noise.{extension}", transparent=True)
    plt.show()
    
    return 0

tim_params = load_params('PAR/cubes.par')
z_list = tim_params['z_list']
dz_list = tim_params['dz_list']
n_list = tim_params['n_list']

for dz, nslice in zip(dz_list, n_list): 
    if(nslice != 2): continue
    b_I(nslice, z_list, (dz,))

