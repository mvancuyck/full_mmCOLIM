from fcts import * 
from gen_all_sizes_cubes import * 
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
    return b, delta_b

def fit_b_eff_auto(angular_k,pk_matter_2d, k, pk_without_sn, sigma_pk, I, delta_I, embed_bool = False ):

    if(embed_bool): embed()
   
    def model_auto_line_to_fit(angular_k,pk_matter_2d, k, I_times_b_co):
        f = interpolate.interp1d(angular_k, pk_matter_2d,  kind='linear')
        pk_matter_nonlin_rebin = f(k)
        return  pk_matter_nonlin_rebin * (I_times_b_co)**2

    F = partial( model_auto_line_to_fit, angular_k.value,  pk_matter_2d)
    if((sigma_pk==0).all()): sigma_pk = np.ones(1e-2)
    popt_J, pcov_J = curve_fit(F, k, pk_without_sn, sigma = sigma_pk, p0 = (I), bounds = (0,(6*I)) )#maxfev=20000,
    bI = popt_J[0]
    delta_bI = np.sqrt(pcov_J[0][0])
    b = bI / I
    delta_b = b * np.sqrt( (delta_I/I)**2 + (delta_bI/bI)**2  ) 
    return b, delta_b

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
                    bgal,dbgal= fit_b_gal(     dict["k_angular"], dict["pk_matter_2d"], k, Gal, sigma_Gal) 
                    bX, dbX   = fit_b_eff_auto(dict["k_angular"], dict["pk_matter_2d"], k, JJ,  sigma_JJ, 
                                               I, sigma_I )
                        
                    b_list[nfield, zi, dzi, j,0,0] = bX
                    b_list[nfield, zi, dzi, j,0,1] = dbX
                    b_list[nfield, zi, dzi, j,1,0] = bgal
                    b_list[nfield, zi, dzi, j,1,1] = dbgal

                    #---------------------
                    if(False): #True and nfield==0 and J=='CO32'):


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
                        axg.set_ylabel("$\\rm P_{gal}$($\\rm k_{\\theta}$) [sr]")
                        axj.set_ylabel("$\\rm P_{"+'{}'.format(J)+"}(k_{\\theta})$ " +"[$\\rm Jy^2/sr$]"); #plt.ylim(5e-6,2e-3)
                        axg.set_xlabel("$\\rm k_{\\theta}$ [$\\rm arcmin^{-1}$]")#, z={z}, dz={dz}")
                        axj.loglog( dict["k"], dict["pk_J"][0].value,
                                   c='darkgray', ls="--", label = 'non-param. $\\rm P^{tot}$(k)' ,lw=lw, zorder=1)
                        
                        axj.errorbar( dict["k"], dict["pk_J"][0].value, yerr=np.std(list_J[:, zi, dzi, j, :], axis=0),
                                   c='darkgray', ls="--", lw=lw)
                        
                        axj.axhline(dict["LIM_shot"][0].value, color='purple', linestyle=':', 
                                   c='darkgray', ls=':',  label = 'shot noise',                   lw=lw, zorder=2 )
                        axj.loglog( dict["k"], dict["pk_J"][0].value - dict["LIM_shot"][0].value, 
                                   c='darkgray', lw=lw,   label = 'non param. $\\rm P^{clust}$(k)',      zorder=3 )
                        axj.loglog( k, JJ, 'or',markersize=mk, label='points used in fit', zorder=5)
                        f = interp1d( dict['k_angular'],  dict['pk_matter_2d'] )
                        axj.set_ylim(2e-5, 4e-1)
                        axj.loglog( k, f(k) * ( bX*I)**2, '--xb',label ='$\\rm (B_{\\nu} \\times b^{\\rm SIDES}_\\mathrm{eff})^2 P^{\\rm 2D}_{\\rm matter}$ \n$ b^{\\rm SIDES}_\\mathrm{eff}$='+f'{np.round(bX,2)}'+'$\\pm$'+f'{np.round(dbX,2)}', lw=lw,markersize=mk, zorder=6)
                        axj.legend(fontsize=BS-2,frameon=False, loc = 'lower left')
                        axg.loglog( dict["k"], dict["pk_gal"][0].value, c='darkgray', ls="--",)
                        axg.errorbar( dict["k"], dict["pk_gal"][0].value, yerr=np.std(list_Gal[:, zi, dzi, j, :], axis=0),  c='darkgray', ls="--",)
                        axg.axhline(dict["gal_shot"][0], linestyle=':', c='darkgray', ls=':', lw=lw, )
                        axg.loglog( dict["k"], dict["pk_gal"][0].value - dict["gal_shot"][0], c='darkgray', lw=lw,)
                        axg.loglog( k, Gal, 'or',markersize=mk)
                        axg.loglog( k, f(k)*bgal**2, '--xg', lw=lw,markersize=mk, 
                                   label ='$ (b^{\\rm SIDES}_\\mathrm{gal})^2 P^{\\rm 2D}_{\\rm matter}$ \n$ b^{\\rm SIDES}_\\mathrm{gal}$='+f'{np.round(bgal,2)}'+'$\\pm$'+f'{np.round(dbgal,2)}')
                        axg.legend(fontsize=BS-2,loc = 'lower left', frameon=False)
                        secax = axj.secondary_xaxis("top", functions=(G,G))
                        secax.set_xlabel('k [$\\rm Mpc^{-1}$]')
                        secax = axj.secondary_yaxis("right", functions=(F,F))
                        secax.set_ylabel("$\\rm P_{"+'{}'.format(J)+"}(k)$ "+' [$\\rm Jy^2/sr^2.Mpc^3]$')
                        secax = axg.secondary_yaxis("right", functions=(F,F))
                        secax.set_ylabel('$\\rm P_{gal}$(k) [$\\rm Mpc^3]$')
                        fig.tight_layout()
                        fig.subplots_adjust(hspace=.0)
                        for extension in ("png", "pdf"): plt.savefig(f"figs/example_{J}_z{z}_dz{dz}_n{nfield}_nslice{n_slices}_fit.{extension}", transparent=True)
                        plt.close()

    return b_list, list_I
 
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
    b_list, I_list = bI_from_autospec_and_cat(n_slices,9,0.15, z_list, dz_list )
    bI_tinker, b_tinker = bI_from_tinker(n_slices,9,0.15, z_list, dz_list)
    bI_mes, bI_mesCO76, bI_mesCI = bI_crosspec(n_slices,9,0.15, z_list, dz_list, b_list,dtype = '_with_interlopers')
    idz=0
    
    patchs=[]
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(5,5), dpi=200); 
    mks=6; dr = 0.04; dy = 0.0; text_y_offset=0.2; lw=1; width = [1,1];  height = [1,1,1,1,1]
    drelative = (-4,-3,-2,-1,0,1,2,3)
    gs = gridspec.GridSpec(ncols=2, nrows=5, height_ratios= height, width_ratios=width)
    axr = plt.subplot(gs[-2]); patchs = []
    axr.set_xlabel(r"redshift")
    axr.set_ylabel("relative \n difference")
    axr.set_ylim(-0.3, 1.2)
    for j, (line, rest_freq) in enumerate(zip(line_list_fancy[:8], rest_freq_list[:8])):
        c=colors_co[j]; 
        #---------------------------
        ax = plt.subplot(gs[j]); 
        #---------------------------
        if(j==7): patch = mpatches.Patch(color='grey', label=r'from cross-power spectrum estimate' ); patchs.append(patch)
        ax.errorbar(    z_list, np.mean(bI_mes[:,:,idz,j], axis=0),  linestyle='solid', color=c, ecolor=c, lw=lw) 
        ax.fill_between(z_list, np.mean(bI_mes[:,:,idz,j], axis=0)-np.std(bI_mes[:,:,idz,j], axis=0), np.mean(bI_mes[:,:,idz,j], axis=0)+np.std(bI_mes[:,:,idz,j], axis=0), alpha = 0.2, color=c)
        if(j==7): patch = mlines.Line2D([], [], color='k', linestyle=":", marker="*",  label='catalogue mean brightness \ntimes effective bias'); patchs.append(patch)
        ax.errorbar( z_list, np.mean(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0), yerr=np.std(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0),
                     linestyle=":", color='k', ecolor='k', marker='*',markersize = mks,lw=lw )
        if(j==7): patch = mlines.Line2D([], [], color='k', linestyle="dashdot", marker="p", mfc='none', label=r'Tinker et al. 2010'); patchs.append(patch)
        ax.errorbar( z_list, np.mean(bI_tinker[:, :, idz, j], axis=0), yerr=np.std(bI_tinker[:, :, idz, j], axis=0),
                     linestyle="dashdot", color='k', ecolor='k', marker="p",markersize = mks,lw=lw,  mfc='none', mec='k')
        if(j==7): ax.legend(handles = patchs, loc='center left', bbox_to_anchor=(-0.4, -0.7), fontsize=7, frameon=False)
        axr.errorbar( np.asarray(z_list)+drelative[j]*dr, (np.mean(bI_mes[:,:,idz,j], axis=0)/np.mean(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0))-1+dy, 
                     yerr = (np.std(bI_mes[:,:,idz,j], axis=0)/np.mean(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0)), linestyle="None", color=c,  ecolor=c, marker='*', markersize = mks, lw=lw )
        #print(f'Mean uncertainty of J={j+1}:'+f'{100*np.mean( sigma_bI[:,j,0]/bI[:,j,0] )}'+'%')
        #print(f'Mean diff with intrinsec of J={j+1}:'+f'{100*np.mean( bI[:,j,0]/bIcons[:,j,0]-1)}'+'%')
        #print('')
        #---------------------------
        ax.set_ylabel(r"$b_{\rm eff} \times$"+"$\\rm B^{"+'{}'.format(line)+"}_{\\nu}$ ")
        ax.set_xlim(0.4, np.max(z_list)+0.1)
        ax.set_yscale("log")
        if(j!=7): ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=False)
        else:
            ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=True)
        if(j == 7): ax.set_xlabel(r"redshift")
    fig.tight_layout(); fig.subplots_adjust(hspace=.0)
    #---------------------------
    for extension in ("png", "pdf"): plt.savefig(f"bI_CO_nslice{nslice}_dz{dz_list[0]}.{extension}", transparent=True)
    plt.show()

    BS = 11; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(3,4.5), dpi=200); lw=1; mks=6; j=6; c=colors_co[6]

    plt.errorbar(    z_list, np.mean(bI_mes[:,:,idz,-2], axis=0),  linestyle='solid', color=colors_co[-2], ecolor=c, lw=lw,  label="CO(7-6) cross-power,\n with interlopers",) 
    plt.fill_between(z_list, np.mean(bI_mes[:,:,idz,-2], axis=0)-np.std(bI_mes[:,:,idz,-2], axis=0), np.mean(bI_mes[:,:,idz,-2], axis=0)+np.std(bI_mes[:,:,idz,-2], axis=0), alpha = 0.2, color=colors_co[-2])
    plt.errorbar(    z_list, np.mean(b_list[:, :, idz, -2,0,0]*I_list[:,:,idz,-2], axis=0), yerr=np.std(b_list[:, :, idz, -2,0,0]*I_list[:,:,idz,-2], axis=0), 
                 label = 'catalogue mean brightness \ntimes fitted effective bias',
                 linestyle=":", color='k', ecolor='k', marker='*',markersize = mks,lw=lw )
    c='grey'
    plt.errorbar(  z_list, np.mean(bI_mesCO76[:,:, idz],axis=0), linestyle='--', color=c, ecolor=c,  label='CO(7-6) cross-power,\n without interlopers', lw=lw) 
    plt.fill_between(z_list, np.mean(bI_mesCO76[:,:, idz],axis=0) - np.std(bI_mesCO76[:,:, idz],axis=0), np.mean(bI_mesCO76[:,:, idz],axis=0) +  np.std(bI_mesCO76[:,:, idz],axis=0), alpha = 0.2, color=c,)
    c='purple'
    plt.errorbar( z_list, np.mean(bI_mesCI[:,:, idz],axis=0),  linestyle='--', color=c, ecolor=c,  label='CI(2-1) Cross-power', lw=lw) # yerr = sigma_bI_all_subfiels[:,j,0],
    plt.fill_between(z_list, np.mean(bI_mesCI[:,:, idz],axis=0) - np.std(bI_mesCI[:,:, idz],axis=0), np.mean(bI_mesCI[:,:, idz],axis=0) +  np.std(bI_mesCI[:,:, idz],axis=0), alpha = 0.2, color=c,)
    
    #plt.errorbar(z_list, mean_cross, yerr = sigma_sum_cross, ls='--',c='dimgray',lw=lw, label='(CO76+[CI]21) cross-power)')    
    line='CO(7-6)'
    plt.ylabel(r"$b_{\rm eff} \times$"+"$\\rm B^{"+'{}'.format(line)+"}_{\\nu}$ ")
    plt.legend(fontsize=6, loc='lower right' , frameon=False)
    plt.xlim(0.4, np.max(z_list)+0.1); 
    plt.yscale("log")
    plt.xlabel(r"redshift")
    plt.tight_layout()
    for extension in ("png", "pdf"): plt.savefig(f"CO76.{extension}") 
    
def rho_LCO(n_slices, z_list, dz_list, ref = 'CO32',  alphaco=3.6, dtype='all_lines', recompute=False):

    colors_co = ('orange', 'r', 'b', 'cyan', 'g', 'purple', 'magenta', 'grey',)
    b_list, I_list = bI_from_autospec_and_cat(n_slices,9,0.15, z_list, dz_list )
    bI_tinker, b_tinker = bI_from_tinker(n_slices,9,0.15, z_list, dz_list)
    bI_mes, bI_mesCO76, bI_mesCI = bI_crosspec(n_slices,9,0.15, z_list, dz_list, b_list,dtype = '_with_interlopers')
    idz=0
    
    patchs=[]
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(5,5), dpi=200); 
    mks=6; dr = 0.04; dy = 0.0; text_y_offset=0.2; lw=1; width = [1,1];  height = [1,1,1]
    drelative = (-4,-3,-2,-1,0,1,2,3)
    gs = gridspec.GridSpec(ncols=2, nrows=3, height_ratios= height, width_ratios=width)
    #axr = plt.subplot(gs[-2]); patchs = []
    #axr.set_xlabel(r"redshift")
    #axr.set_ylabel("relative \n difference")
    #axr.set_ylim(-0.3, 1.2)
    for j, (line, rest_freq) in enumerate(zip(line_list_fancy[:6], rest_freq_list[:6])):
        c=colors_co[j]; 
        #---------------------------
        ax = plt.subplot(gs[j]); 
        #---------------------------
        if(j==7): patch = mpatches.Patch(color='grey', label=r'from cross-power spectrum estimate' ); patchs.append(patch)

        dict = pickle.load( open('rho_9deg2.p', 'rb'))
        ax.errorbar( z_list, np.mean(dict['rho_9deg2'], axis=1), linestyle="--", color='grey', ecolor='grey', markersize = mks,lw=lw,  mfc='none', mec='k')
        ax.fill_between( z_list, np.mean(dict['rho_9deg2'], axis=1)-np.std(dict['rho_9deg2'], axis=1), np.mean(dict['rho_9deg2'], axis=1)+np.std(dict['rho_9deg2'], axis=1),
                            color='grey', alpha=0.3)


        B = bI_mes[:,:,idz,j]/b_list[:, :, idz, j,0,0]
        if(j==0): 
            excitation = B
            excitation_ref = bI_mes[:,:,idz,2]/b_list[:, :, idz, 2,0,0]
            EXCITATION_ASSUMED = (excitation/excitation_ref)
        #if(j!=0): EXCITATION_ASSUMED = (excitation/excitation_ref).mean()
        z_list = np.asarray(z_list)
        rhoL = ((4*np.pi*115.27120180e9*cosmo.H(z_list))/(4e7 *cst.c*1e-3)).value #Lsolar/Mpc3
        nu_obs = 115.27120180 / (1+z_list)
        Lprim = 3.11e10/(nu_obs*(1+z_list))**3
        rhoh2 = EXCITATION_ASSUMED * excitation_ref/B * rhoL *alphaco*Lprim#if(j==7): patch = mlines.Line2D([], [], color='k', linestyle=":", marker="*",  label='catalogue mean brightness \ntimes effective bias'); patchs.append(patch)
        #ax.errorbar( z_list, np.mean(I_list[:,:,idz,j]*rhoh2, axis=0), yerr=np.std(I_list[:,:,idz,j]*rhoh2, axis=0),
        #             linestyle=":", color='k', ecolor='k', marker='*',markersize = mks,lw=lw )


        #ax.errorbar(    z_list, np.mean(B*rhoh2, axis=0),  linestyle='solid', color='magenta', ecolor='magenta', lw=lw) 
        #ax.fill_between(z_list, np.mean(B*rhoh2, axis=0)-np.std(B*rhoh2, axis=0), np.mean(B*rhoh2, axis=0)+np.std(B*rhoh2, axis=0),
        #                alpha = 0.2, color='magenta')
        
        if(j==7): ax.legend(handles = patchs, loc='center left', bbox_to_anchor=(-0.4, -0.7), fontsize=7, frameon=False)

        B = bI_mes[:,:,idz,j]/b_tinker[:, :, idz, j]
        ax.errorbar(    z_list, np.mean(B*rhoh2, axis=0), linestyle="dashdot", color='k', ecolor='k', marker="p",markersize = mks,lw=lw,  mfc='none', mec='k')
        #axr.errorbar( np.asarray(z_list)+drelative[j]*dr, (np.mean(bI_mes[:,:,idz,j], axis=0)/np.mean(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0))-1+dy, 
        #             yerr = (np.std(bI_mes[:,:,idz,j], axis=0)/np.mean(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0)), linestyle="None", color=c,  ecolor=c, marker='*', markersize = mks, lw=lw )
        #print(f'Mean uncertainty of J={j+1}:'+f'{100*np.mean( sigma_bI[:,j,0]/bI[:,j,0] )}'+'%')
        #print(f'Mean diff with intrinsec of J={j+1}:'+f'{100*np.mean( bI[:,j,0]/bIcons[:,j,0]-1)}'+'%')
        #print('')
        #---------------------------
        #ax.set_ylabel("$\\rm B^{"+'{}'.format(line)+"}_{\\nu}$")#+'$\\rm [Jy/sr]$')
        #ax.set_ylabel("$\\rm \\rho_L^{"+'{}'.format(line_list_fancy[0])+"}$"+f'\n from {line}')#+'$\\rm [Jy/sr]$')
        ax.set_ylabel("$\\rm \\rho_{H2} \, [M_{\\odot}.Mpc^{-3}]$"+f'\n from {line}')#+'$\\rm [Jy/sr]$')

        ax.set_xlim(0.4, np.max(z_list)+0.1)
        ax.set_yscale("log")
        if(j<4): ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=False)
        else:
            ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=True)
        if(j == 4 or j==5): ax.set_xlabel(r"redshift")

    patch = mlines.Line2D([], [], color='grey', linestyle="--",  label="$\\rm \\rho_{L'CO(1-0)} \\times \\alpha_{CO} $"); patchs.append(patch)
    patch = mpatches.Patch(color='grey', label='from cross-power and $\\rm b^{SIDES}_{eff}$' ); patchs.append(patch)
    #patch = mlines.Line2D([], [], color='k', linestyle=":", marker="*",  label='catalogue mean brightness \ntimes effective bias'); patchs.append(patch)
    patch = mlines.Line2D([], [], color='k', linestyle="dashdot", marker="p", mfc='none', label="from cross-power and $\\rm b^{Tinker+2010}_{eff}$"); patchs.append(patch)
    ax.legend(handles = patchs, loc='center left', bbox_to_anchor=(-0.3, -0.5), fontsize=7, frameon=False)

    fig.tight_layout(); fig.subplots_adjust(hspace=.0)
    #---------------------------
    for extension in ("png", "pdf"): plt.savefig(f"bI_CO_nslice{nslice}_dz{dz_list[0]}.{extension}", transparent=True)
    plt.show()

def minib_I(n_slices, z_list, dz_list, ref = 'CO32',  dtype='all_lines', recompute=False):

    colors_co = ('orange', 'r', 'b', 'cyan', 'g', 'purple', 'magenta', 'grey',)
    b_list, I_list = bI_from_autospec_and_cat(n_slices,9,0.15, z_list, dz_list )
    bI_tinker = bI_from_tinker(n_slices,9,0.15, z_list, dz_list)
    bI_mes, bI_mesCO76, bI_mesCI = bI_crosspec(n_slices,9,0.15, z_list, dz_list, b_list,dtype = '_with_interlopers')
    idz=0
    
    patchs=[]
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(5,5), dpi=200); 
    mks=6; dr = 0.04; dy = 0.0; text_y_offset=0.2; lw=1; width = [1,1];  height = [1,1,1,1]
    drelative = (-4,-3,-2,-1,0,1,2,3)
    gs = gridspec.GridSpec(ncols=2, nrows=4, height_ratios= height, width_ratios=width)
    patchs = []

    for j, (line, rest_freq) in enumerate(zip(line_list_fancy[:8], rest_freq_list[:8])):
        c=colors_co[j]; 
        #---------------------------
        ax = plt.subplot(gs[j]); 
        #---------------------------
        if(j==7): patch = mpatches.Patch(color='grey', label=r'from cross-power spectrum estimate' ); patchs.append(patch)
        ax.errorbar(    z_list, np.mean(bI_mes[:,:,idz,j], axis=0),  linestyle='solid', color=c, ecolor=c, lw=lw) 
        ax.fill_between(z_list, np.mean(bI_mes[:,:,idz,j], axis=0)-np.std(bI_mes[:,:,idz,j], axis=0), np.mean(bI_mes[:,:,idz,j], axis=0)+np.std(bI_mes[:,:,idz,j], axis=0), alpha = 0.2, color=c)
        if(j==7): patch = mlines.Line2D([], [], color='k', linestyle=":", marker="*",  label='catalogue mean brightness \ntimes effective bias'); patchs.append(patch)
        ax.errorbar( z_list, np.mean(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0), yerr=np.std(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0),
                     linestyle=":", color='k', ecolor='k', marker='*',markersize = mks,lw=lw )
        #if(j==7): patch = mlines.Line2D([], [], color='k', linestyle="dashdot", marker="p", mfc='none', label='$\\rm \\Sigma$ S$\\rm \\times b^{Tinker. 2010}_{eff}$/$\\rm \\Omega_{field}$'); patchs.append(patch)
        #ax.errorbar( z_list, np.mean(bI_tinker[:, :, idz, j], axis=0), yerr=np.std(bI_tinker[:, :, idz, j], axis=0),
        #             linestyle="dashdot", color='k', ecolor='k', marker="p",markersize = mks,lw=lw,  mfc='none', mec='k')
        if(j==7): ax.legend(handles = patchs, loc='center left', bbox_to_anchor=(-0.4, -0.7), fontsize=7, frameon=False)


        print('')
        print('')
        print(line, 100*np.std(bI_mes[:,:,idz,j], axis=0)/np.mean(bI_mes[:,:,idz,j], axis=0))
        print('')
        print('')

        #print(f'Mean uncertainty of J={j+1}:'+f'{100*np.mean( sigma_bI[:,j,0]/bI[:,j,0] )}'+'%')
        #print(f'Mean diff with intrinsec of J={j+1}:'+f'{100*np.mean( bI[:,j,0]/bIcons[:,j,0]-1)}'+'%')
        #print('')
        #---------------------------
        ax.set_ylabel(r"$b_{\rm eff} \times$"+"$\\rm B^{"+'{}'.format(line)+"}_{\\nu}$ ")
        ax.set_xlim(0.4, np.max(z_list)+0.1)
        ax.set_ylim(5e1, 7e2)
        ax.set_yscale("log")
        if(j!=7): ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=False)
        else:
            ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=True)
        if(j == 7 or j==6): ax.set_xlabel(r"redshift")
    fig.tight_layout(); fig.subplots_adjust(hspace=.0)
    #---------------------------
    for extension in ("png", "pdf"): plt.savefig(f"bI_CO_nslice{nslice}_dz{dz_list[0]}.{extension}", transparent=True)
    plt.show()

def b_comp(n_slices, z_list, dz_list, ref = 'CO32',  dtype='all_lines', recompute=False):

    colors_co = ('orange', 'r', 'b', 'cyan', 'g', 'purple', 'magenta', 'grey',)
    b_list, I_list = bI_from_autospec_and_cat(n_slices,9,0.15, z_list, dz_list )
    bI_tinker, b_tinker = bI_from_tinker(n_slices,9,0.15, z_list, dz_list)
    bI_mes, bI_mesCO76, bI_mesCI = bI_crosspec(n_slices,9,0.15, z_list, dz_list, b_list,dtype = '_with_interlopers')
    idz=0; mks=7; lw=1
    
    patchs=[]
    BS = 12; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    plt.figure(figsize=(3,3), dpi=200); 
    j=0

    c=colors_co[j]; 
    #---------------------------
    #ax.errorbar(    z_list, np.mean(bI_mes[:,:,idz,j], axis=0),  linestyle='solid', color=c, ecolor=c, lw=lw) 
    #ax.fill_between(z_list, np.mean(bI_mes[:,:,idz,j], axis=0)-np.std(bI_mes[:,:,idz,j], axis=0), np.mean(bI_mes[:,:,idz,j], axis=0)+np.std(bI_mes[:,:,idz,j], axis=0), alpha = 0.2, color=c)

    plt.errorbar( z_list, np.mean(b_list[:, :, idz, j,0,0], axis=0), yerr=np.std(b_list[:, :, idz, j,0,0], axis=0),
                    linestyle=":", color='k', ecolor='k', marker='*',markersize = mks,lw=lw )
    plt.errorbar( z_list, np.mean(b_tinker[:, :, idz, j], axis=0), yerr=np.std(b_tinker[:, :, idz, j], axis=0),
                    linestyle="dashdot", color='k', ecolor='k', marker="p",markersize = mks,lw=lw,  mfc='none', mec='k')
    plt.ylabel("$\\rm b_{eff}$")
    plt.xlim(0.4, np.max(z_list)+0.1)
    plt.xlabel("redshift")
    
    patch = mlines.Line2D([], [], color='k', linestyle=":", marker="*", label=r'fit from auto-power' ); patchs.append(patch)
    patch = mlines.Line2D([], [], color='k', linestyle="dashdot", marker="p", mfc='none', label=r'Tinker et al. 2010'); patchs.append(patch)
    plt.legend(handles = patchs, loc='upper left', fontsize=7, frameon=False)
    plt.tight_layout()
    #---------------------------
    for extension in ("png", "pdf"): plt.savefig(f"bI_CO_nslice{nslice}_dz{dz_list[0]}.{extension}", transparent=True)
    plt.show()
#With SIDES Bolshoi, for rapid tests. 

def mini_rho(n_slices, z_list, dz_list, ref = 'CO32',  dtype='all_lines', recompute=False, alphaco=3.6):

    colors_co = ('orange', 'r', 'b', 'cyan', 'g', 'purple', 'magenta', 'grey',)
    b_list, I_list = bI_from_autospec_and_cat(n_slices,9,0.15, z_list, dz_list )
    bI_tinker, b_tinker = bI_from_tinker(n_slices,9,0.15, z_list, dz_list)
    bI_mes, bI_mesCO76, bI_mesCI = bI_crosspec(n_slices,9,0.15, z_list, dz_list, b_list,dtype = '_with_interlopers')
    idz=0
    
    patchs=[]
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(5,5), dpi=200); 
    mks=6; dr = 0.04; dy = 0.0; text_y_offset=0.2; lw=1; width = [1,1];  height = [1,1,1]
    drelative = (-4,-3,-2,-1,0,1,2,3)
    gs = gridspec.GridSpec(ncols=2, nrows=3, height_ratios= height, width_ratios=width)
    patchs = []

    for j, (line, rest_freq) in enumerate(zip(line_list_fancy[:6], rest_freq_list[:6])):
        c=colors_co[j]; 
        #---------------------------
        ax = plt.subplot(gs[j]); 
        #-------------------------
        dict_of_excitation = pickle.load( open(f'dict_dir/SLED_mes_9deg2_dtype_with_interlopers.p', 'rb'))
        SLED_mes = dict_of_excitation['SLED_mes']
        excitation_MES = SLED_mes[:,:,idz, j+1]
        #-------------------------
        B = bI_mes[:,:,idz,j]/b_list[:, :, idz, j,0,0]
        Btinker = bI_mes[:,:,idz,j]/b_tinker[:, :, idz, j]
        #-------------------------
        if(j<4): ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=False)
        else:
            ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=True)
        #-------------------------
        if(j==0): 
            ax.tick_params(axis = "y", which='major',left = False, labelleft=False)
            excitation =     bI_mes[:,:,idz,0]/b_list[:, :, idz, 0,0,0]
            excitation_ref = bI_mes[:,:,idz,2]/b_list[:, :, idz, 2,0,0]
            EXCITATION_ASSUMED = (excitation/excitation_ref)
            continue
        #-------------------------
        z_list = np.asarray(z_list)
        rhoL = ((4*np.pi*115.27120180e9*cosmo.H(z_list))/(4e7 *cst.c*1e-3)).value #Lsolar/Mpc3
        nu_obs = 115.27120180 / (1+z_list)
        Lprim = 3.11e10/(nu_obs*(1+z_list))**3
        rhoh2 =  rhoL *alphaco*Lprim
        #-------------------------
        ax.errorbar(    z_list, np.mean(B/excitation_MES*EXCITATION_ASSUMED*rhoh2, axis=0),  linestyle='solid', color=c, ecolor=c, lw=lw) 
        ax.fill_between(z_list, 
                        np.mean(B/excitation_MES*EXCITATION_ASSUMED*rhoh2, axis=0)-np.std(B/excitation_MES*EXCITATION_ASSUMED*rhoh2, axis=0), 
                        np.mean(B/excitation_MES*EXCITATION_ASSUMED*rhoh2, axis=0)+np.std(B/excitation_MES*EXCITATION_ASSUMED*rhoh2, axis=0), 
                        alpha = 0.2, color=c)
        #-------------------------
        ax.errorbar(z_list, np.mean(Btinker/excitation_MES*EXCITATION_ASSUMED*rhoh2, axis=0), yerr=np.std(Btinker/excitation_ref*EXCITATION_ASSUMED*rhoh2, axis=0),  
                    linestyle="dashdot", color='k', ecolor='k', marker="p",markersize = mks,lw=lw,  mfc='none', mec='k')
        #-------------------------
        dict = pickle.load( open('rho_9deg2.p', 'rb'))
        ax.errorbar( z_list, np.mean(dict['rho_9deg2'], axis=1), linestyle="--", color='grey', ecolor='grey', markersize = mks,lw=lw,  mfc='none', mec='k')
        ax.fill_between( z_list, np.mean(dict['rho_9deg2'], axis=1)-np.std(dict['rho_9deg2'], axis=1), np.mean(dict['rho_9deg2'], axis=1)+np.std(dict['rho_9deg2'], axis=1),
                            color='grey', alpha=0.3)
        ax.set_ylim(3e7, 9e7)
        #---------------------------
        #ax.set_ylabel(r"$b_{\rm eff} \times$"+"$\\rm B^{"+'{}'.format(line)+"}_{\\nu}$ ")
        #ax.set_ylabel(r"$b_{\rm eff} \times$"+"$\\rm B^{C0(3-2)}_{\\nu}$ \n from"+f" {line}")
        #ax.set_ylabel("$\\rm B^{C0(3-2)}_{\\nu}$ \n from"+f" {line}")
        #ax.set_ylabel("$\\rm B^{C0(1-0)}_{\\nu}$ \n from"+f" {line}")
        ax.set_ylabel("$\\rm \\rho_{H2} \, [M_{\\odot}.Mpc^{-3}]$"+f'\n from {line}')#+'$\\rm [Jy/sr]$')
        ax.set_xlim(0.4, np.max(z_list)+0.1)
        ax.set_yscale("log")
        if(j == 4 or j==5): ax.set_xlabel(r"redshift")
        #-------------------------
    patch = mlines.Line2D([], [], color='grey', linestyle="--",  label="$\\rm \\rho_{L'CO(1-0)} \\times \\alpha_{CO} $"); patchs.append(patch)
    patch = mpatches.Patch(color='grey', label='from cross-power / $\\rm b^{SIDES}_{eff}$' ); patchs.append(patch)
    patch = mlines.Line2D([], [], color='k', linestyle="dashdot", marker="p", mfc='none', label="from cross-power / $\\rm b^{Tinker+2010}_{eff}$"); patchs.append(patch)
    ax.legend(handles = patchs, loc='center left', bbox_to_anchor=(-0.4, -0.5), fontsize=8, frameon=False)
    #-------------------------
    fig.tight_layout(); fig.subplots_adjust(hspace=.0)
    #---------------------------
    for extension in ("png", "pdf"): plt.savefig(f"minirho_nslice{nslice}_dz{dz_list[0]}.{extension}", transparent=True)
    plt.show()

def rho_excitation(n_slices, z_list, dz_list, ref = 'CO32',  dtype='all_lines', recompute=False, alphaco=3.6):

    colors_co = ('orange', 'r', 'b', 'cyan', 'g', 'purple', 'magenta', 'grey',)
    b_list, I_list = bI_from_autospec_and_cat(n_slices,9,0.15, z_list, dz_list )
    bI_tinker, b_tinker = bI_from_tinker(n_slices,9,0.15, z_list, dz_list)
    bI_mes, bI_mesCO76, bI_mesCI = bI_crosspec(n_slices,9,0.15, z_list, dz_list, b_list,dtype = '_with_interlopers')
    idz=0
    
    patchs=[]
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(5,5), dpi=200); 
    mks=6; dr = 0.04; dy = 0.0; text_y_offset=0.2; lw=1; width = [1,1];  height = [1,1,1]
    drelative = (-4,-3,-2,-1,0,1,2,3)
    gs = gridspec.GridSpec(ncols=2, nrows=3, height_ratios= height, width_ratios=width)
    patchs = []

    for j, (line, rest_freq) in enumerate(zip(line_list_fancy[:6], rest_freq_list[:6])):
        c=colors_co[j]; 
        #---------------------------
        ax = plt.subplot(gs[j]); 
        if(j<4): ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=False)
        else:
            ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=True)
        
        #-------------------------
        B = bI_mes[:,:,idz,j]/b_list[:, :, idz, j,0,0]
        Btinker = bI_mes[:,:,idz,j]/b_tinker[:, :, idz, j]
        if(j==0): 
            ax.tick_params(axis = "y", which='major',left = False, labelleft=False)
            #-------------------------
            excitation =     bI_mes[:,:,idz,0]/b_list[:, :, idz, 0,0,0]
            excitation_ref = bI_mes[:,:,idz,2]/b_list[:, :, idz, 2,0,0]
            EXCITATION_ASSUMED = (excitation/excitation_ref)
            
            continue
    

        dict_of_excitation = pickle.load( open(f'dict_dir/SLED_mes_9deg2_dtype_with_interlopers.p', 'rb'))
        SLED_mes = dict_of_excitation['SLED_mes']
        excitation_MES = SLED_mes[:,:,idz, j+1]
        #excitation_MES = B / excitation_ref
        #-------------------------
        z_list = np.asarray(z_list)
        rhoL = ((4*np.pi*115.27120180e9*cosmo.H(z_list))/(4e7 *cst.c*1e-3)).value #Lsolar/Mpc3
        nu_obs = 115.27120180 / (1+z_list)
        Lprim = 3.11e10/(nu_obs*(1+z_list))**3
        rhoh2 =  rhoL *alphaco*Lprim
        #-------------------------
        ax.errorbar(    z_list, np.mean(B/excitation_MES*rhoh2, axis=0),  linestyle='solid', color=c, ecolor=c, lw=lw) 
        ax.fill_between(z_list, 
                        np.mean(B/excitation_MES*rhoh2, axis=0)-np.std(B/excitation_MES*rhoh2, axis=0), 
                        np.mean(B/excitation_MES*rhoh2, axis=0)+np.std(B/excitation_MES*rhoh2, axis=0), 
                        alpha = 0.2, color=c)
        #-------------------------
        print('')
        print('')
        print(line, 100*np.std(B/excitation_MES*rhoh2, axis=0)/np.mean(B/excitation_MES*rhoh2, axis=0) )
        print('')
        print('')
        ax.errorbar(z_list, np.mean(Btinker/excitation_MES*rhoh2, axis=0), yerr=np.std(Btinker/excitation_ref*EXCITATION_ASSUMED*rhoh2, axis=0),  
                    linestyle="dashdot", color='k', ecolor='k', marker="p",markersize = mks,lw=lw,  mfc='none', mec='k')
        #-------------------------
        dict = pickle.load( open('rho_9deg2.p', 'rb'))
        RHO = np.transpose(dict['rho_9deg2'])
        ax.errorbar( z_list, np.mean(RHO/EXCITATION_ASSUMED, axis=0), linestyle="--", color='grey', ecolor='grey', markersize = mks,lw=lw,  mfc='none', mec='k')
        ax.fill_between( z_list, 
                        np.mean(RHO/EXCITATION_ASSUMED, axis=0)-np.std(RHO/EXCITATION_ASSUMED, axis=0), 
                        np.mean(RHO/EXCITATION_ASSUMED, axis=0)+np.std(RHO/EXCITATION_ASSUMED, axis=0),
                            color='grey', alpha=0.3)
        
        dict = pickle.load( open('rho_9deg2.p', 'rb'))
        ax.errorbar( z_list, np.mean(dict['rho_9deg2'], axis=1), linestyle="--", color='lightgreen', ecolor='lightgreen', markersize = mks,lw=lw,  mfc='none', mec='k')
        ax.fill_between( z_list, 
                        np.mean(dict['rho_9deg2'], axis=1)-np.std(dict['rho_9deg2'], axis=1), 
                        np.mean(dict['rho_9deg2'], axis=1)+np.std(dict['rho_9deg2'], axis=1),
                            color='lightgreen', alpha=0.3)
        #---------------------------
        ax.set_ylabel("$\\rm \\rho_{H2}  \\times R^{1-ref=3}$"+f'\n from {line}')#+'$\\rm [Jy/sr]$')
        ax.set_xlim(0.4, np.max(z_list)+0.1)
        ax.set_yscale("log")
        ax.set_ylim(4e7, 4e8)
        if(j == 4 or j==5): ax.set_xlabel(r"redshift")
        #-------------------------
    patch = mlines.Line2D([], [], color='lightgreen', linestyle="--",  label="$\\rm \\rho_{L'CO(1-0)} \\times \\alpha_{CO} $"); patchs.append(patch)
    patch = mlines.Line2D([], [], color='grey', linestyle="--",  label="$\\rm \\rho_{L'CO(1-0)} \\times \\alpha_{CO} \\times R^{1-ref=3} $"); patchs.append(patch)
    patch = mpatches.Patch(color='grey', label='from cross-power / $\\rm b^{SIDES}_{eff}$' ); patchs.append(patch)
    patch = mlines.Line2D([], [], color='k', linestyle="dashdot", marker="p", mfc='none', label="from cross-power / $\\rm b^{Tinker+2010}_{eff}$"); patchs.append(patch)
    ax.legend(handles = patchs, loc='center left', bbox_to_anchor=(-0.5, -0.5), fontsize=8, frameon=False)
    #-------------------------
    fig.tight_layout(); fig.subplots_adjust(hspace=.0)
    #---------------------------
    for extension in ("png", "pdf"): plt.savefig(f"excitation_rho_nslice{nslice}_dz{dz_list[0]}.{extension}", transparent=True)
    plt.show()

def ngal_vs_z(n_slices, z_list, dz_list,):


    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(2,2), dpi=200); 
    for dzi, dz in enumerate(dz_list):
        for zi, z in enumerate(z_list):
            ngals = []
            for nfield in range(12):

                file = f"dict_dir/dict_LIMgal_pySIDES_from_uchuu_ntile_{nfield}_z{z}_dz{dz}_{n_slices}slices_9deg2_CO10.p"
                d = pickle.load( open(file, 'rb'))
                ngals.append( 1/(d["gal_shot"][0])*(9*u.deg**2).to(u.sr).value )

            plt.errorbar(z, np.asarray(ngals).mean(), np.asarray(ngals).std(), fmt='o', color='k', ecolor='k', markersize=2, lw=1)
    plt.yscale('log')
    plt.ylabel('Nb of galaxies')
    plt.xlabel('redshift')
    plt.tight_layout()
    plt.savefig('ngal_vs_z.png', transparent=True)
    plt.show()


tim_params = load_params('PAR/cubes.par')
z_list = tim_params['z_list']
dz_list = tim_params['dz_list']
n_list = tim_params['n_list']

for dz, nslice in zip(dz_list, n_list): 
    #minib_I(nslice, z_list, (dz,))
    #mini_rho(nslice, z_list, (dz,))
    
    #b_comp(nslice, z_list, (dz,))
    #rho_excitation(nslice, z_list, (dz,))
    ngal_vs_z(nslice, z_list, (dz,))

