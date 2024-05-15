from fct_co_paper import * 
from gen_cube_co_paper import * 
from functools import partial
from multiprocessing import Pool, cpu_count
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
import argparse
#plt.ion()
"""
def compute_dA(A,B,dB,C,dC, includenewaxis = False):
    if(includenewaxis): return np.sqrt((dB/B)**2+( (dC/C)[:,:,np.newaxis] )**2)*A
    else: return np.sqrt((dB/B)**2+(dC/C)**2)*A
"""

def fit_b_gal(angular_k,pk_matter_2d, k, pk_without_sn, sigma_pk ):
    
    def model_auto_gal_to_fit(angular_k,pk_matter_2d, k, b):
        f = interpolate.interp1d(angular_k, pk_matter_2d,  kind='linear')
        pk_matter_nonlin_rebin = f(k)
        return  pk_matter_nonlin_rebin * b**2
    
    F = partial( model_auto_gal_to_fit, angular_k.value,  pk_matter_2d)
    if((sigma_pk==0).all()): sigma_pk = np.ones(1e-5)
    popt_J, pcov_J = curve_fit(F, k, pk_without_sn, sigma = sigma_pk, p0 = (1), bounds = (0,(6)) )#maxfev=20000,
    b = popt_J[0]; delta_b = np.sqrt(pcov_J[0][0])
    return b, delta_b

def fit_b_eff_auto(angular_k,pk_matter_2d, k, pk_without_sn, sigma_pk, I, delta_I ):
   
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


def bI_from_nsubfields(n_slices, dtype, res, dz,simu,   ksimu, colors, allpoints = False, recompute=False):

    #---------------------------
    patchs = []

    if(simu == 'uchuu'): nfields=1
    else: nfields = 12
    
    bI_mes  = np.zeros((nfields, len(z_list), 8, 1, 2)) 
    bI_cons = np.zeros((nfields, len(z_list), 8, 2, 2))
    bI_cI21 = np.zeros((nfields, len(z_list)))
    pk_mes  = np.zeros((nfields, len(z_list), 8))
    b_mes   = np.zeros((nfields, len(z_list), 8, 2))
    
    for nfield in range(nfields):
        if(simu == 'uchuu'): simuu = 'uchuu';  field_size = 117; cat=None
        else: simuu, cat, _, field_size = load_9deg_subfield(nfield,load_cat=False) 
        
        for z, zi, c in zip(z_list, range(len(z_list)), colors ):
            for j, J, rest_freq in zip(np.arange(8), line_list[:8], rest_freq_list[:8]):

                line_noint = f"{simuu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{field_size}deg2_{J}"
                galaxy = f"{simuu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{field_size}deg2_galaxies_{J}"
                dict_Jg_noint = powspec_LIMgal(line_noint, line_noint, galaxy,output_path, J,  z, dz, n_slices, field_size, dll)
                dict_J = compute_other_linear_model_params( line_noint, line_noint, output_path, J, rest_freq, z, dz, n_slices, field_size, cat, dict_Jg_noint)
                freq_obs = rest_freq/(1+z)
                dnu=dz*freq_obs/(1+z)
                #-----------------                
                #Without interlopers
                if(dtype == None):
                    line =  line_noint
                    dict_Jg = dict_Jg_noint 
                else:
                    line = line_noint + f'_{dtype}'
                    dict_Jg = powspec_LIMgal(line, line, galaxy,output_path, J,  z, dz, n_slices, field_size, dll)
            
                #-----------------                
                k_matter_3D = (dict_Jg["k"].to(u.rad**-1) * 2 * np.pi  /  dict_J["Dc"]).value
                w = np.where( k_matter_3D <= ksimu)
                k_a = w[0][0]; k_e =  w[0][-1]
                if(k_e <= 1):  k_e = 2
                if(allpoints): born = [0,-1]
                else: born = [k_a,k_e+1]
                intK, slopeK = print_scientific_notation( np.mean(k_matter_3D[k_e]) ) #({intK}"+r"$\times$"+rf"$10^{slopeK}$)

                if(nfield ==0): patch = mpatches.Patch(color=c, label=f'z={z}'+"@k$\\leq$"+f"np.round(k,2)"+r"$ \rm Mpc^{-1}$"); patchs.append(patch)
                f = interpolate.interp1d( dict_J['k'], dict_J['pk_matter_2d'],  kind='linear')
                p2d = f(dict_Jg["k"][born[0]:born[1]])
                #-----------------

                I = dict_J["I"][0]; I_sigma = dict_J["I"][1]
                K          = dict_Jg["k"][born[0]:born[1]]
                Gal        = dict_Jg["pk_gal"][0][born[0]:born[1]].value-dict_J["gal_shot"][0]
                sigma_Gal  = dict_Jg["pk_gal"][1][born[0]:born[1]]
                Jgal       = dict_Jg["pk_J-gal"][0][born[0]:born[1]].value - dict_J["LIMgal_shot"][0].value
                sigma_Jgal = dict_Jg["pk_J-gal"][1][born[0]:born[1]]
                Jgal_noint = dict_Jg_noint["pk_J-gal"][0][born[0]:born[1]].value - dict_J["LIMgal_shot"][0].value
                JJ         = dict_Jg_noint["pk_J"][0][born[0]:born[1]].value - dict_J["LIM_shot"][0].value
                sigma_JJ   = dict_Jg_noint["pk_J"][1][born[0]:born[1]].value

                bgal,dbgal= fit_b_gal(     dict_J["k"], dict_J["pk_matter_2d"], K, Gal, sigma_Gal) 
                bX, dbX   = fit_b_eff_auto(dict_J["k"], dict_J["pk_matter_2d"], K, JJ,  sigma_JJ, I.value, I_sigma.value )
                b_mes[nfield, zi, j, 0] = bX; b_mes[nfield, zi, j, 1] = dbX
                
                pk_mes[nfield, zi, j]       =( Jgal).mean()
                bI_mes[nfield, zi, j, 0,0]  =( Jgal / bgal / p2d ).mean()
                bI_mes[nfield, zi, j, 0,1]  =( Jgal / bgal / p2d ).std()
                bI_cons[nfield, zi, j,0 ,0] = bX * I.value 
                bI_cons[nfield, zi, j,1 ,0] = dict_J["beff_t10"][0] * I.value

                #----------------------
                if(False):


                    '''
                    BS=8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS);
                    fig, ax = plt.subplots(1,1,figsize=(3,3), dpi=200); lw=1
                    ax.set_xlabel("k [$\\rm arcmin^{-1}$]")
                    ax.set_ylim(5e-4,1e-2)
                    ax.set_xlim(3e-2, 1e0)
                    ax.set_ylabel(f"P{J}(k) "+"[$\\rm Jy^2$/sr]"); #plt.ylim(5e-6,2e-3)
                    ax.loglog( dict_J["k"].value, (dict_J["pk_matter_2d"]*bX**2*I**2).value+dict_J["LIM_shot"][0].value, c='k', label = '$\\rm P^{tot}$(k)',lw=lw, zorder=1)
                    ax.loglog( dict_J["k"].value, (dict_J["pk_matter_2d"]*bX**2*I**2).value                            , c='r', label = '$\\rm P^{clustering}$(k)',lw=lw, zorder=2)
                    ax.loglog( dict_J["k"].value, np.ones(len(dict_J["k"]))*7e-4, c='b', ls='--',  label = '1-halo', lw=lw, zorder=3 )
                    ax.loglog( dict_J["k"].value, np.ones(len(dict_J["k"]))*dict_J["LIM_shot"][0], c='g', ls=':',  label = 'shot noise', lw=lw, zorder=4 )
                    ax.legend(loc = 'upper right')
                    fig.tight_layout()
                    for extension in ("png", "pdf"): plt.savefig(f"example_pk_model.{extension}", transparent=True)

                    BS=8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS);
                    fig, ax = plt.subplots(1,1,figsize=(3,3), dpi=200); lw=1
                    ax.set_xlabel("k [$\\rm arcmin^{-1}$]")
                    ax.set_ylabel(f"P{J}(k) "+"[$\\rm Jy^2$/sr]"); #plt.ylim(5e-6,2e-3)
                    ax.set_xlim(8e-3, 1)
                    ax.loglog( dict_Jg_noint["k"], dict_Jg_noint["pk_J"][0].value, c='k', label = '$\\rm P^{tot}$(k)',lw=lw, zorder=1)
                    ax.loglog( dict_Jg_noint["k"], dict_Jg_noint["pk_J"][0].value-dict_J["LIM_shot"][0].value, c='r', label = '$\\rm P^{clustering}$(k)',lw=lw, zorder=2)
                    ax.loglog( dict_J["k"],        np.ones(len(dict_J["k"]))*dict_J["LIM_shot"][0].value, c='g', ls=':',  label = 'shot noise', lw=lw, zorder=3 )
                    ax.legend(loc = 'upper right')
                    fig.tight_layout()
                    for extension in ("png", "pdf"): plt.savefig(f"example_pk_model.{extension}", transparent=True)    

                    BS = 7; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); lw = 1; mk=5
                    fig, axs = plt.subplots(1,1, figsize=(3,3), dpi=200)
                    plt.loglog(K, Jgal, '--og', lw=lw, markersize=mk)
                    plt.loglog(K, Jgal_noint, ':xr', lw=lw, markersize=mk)
                    plt.show()

                    '''

                    k_2d_to_3d  = dict_J['Dc']*2*np.pi 
                    pk_2d_to_3d = dict_J['Dc']**2*dict_J['delta_Dc']    
                    def g(k_2d_to_3d, x): return (x*u.arcmin**-1).to(u.rad**-1).value / k_2d_to_3d.value
                    G = partial( g, k_2d_to_3d )
                    def h(pk_2d_to_3d, x): return x * pk_2d_to_3d.value
                    F = partial( h, pk_2d_to_3d )

                    
                    BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); lw = 1; mk=3
                    fig, axs = plt.subplots(1,2, figsize=(6,3), dpi=200)
                    ax=axs[0]
                    ax.set_ylabel(f"P{J}(k) "+"[$\\rm Jy^2$/sr]"); #plt.ylim(5e-6,2e-3)
                    ax.set_xlabel("k [$\\rm arcmin^{-1}$]")
                    ax.set_xlim(8e-3, 1)
                    ax.set_ylim(2e-5, 8e-2)
                    ax.loglog( dict_Jg_noint["k"],        dict_Jg_noint["pk_J"][0].value,                                  'darkgray', ls="--", label = 'non-param. $\\rm P^{tot}$(k)' ,lw=lw, zorder=1)
                    ax.loglog( dict_J["k"],               np.ones(len(dict_J["k"]))*dict_J["LIM_shot"][0].value,         c='darkgray', ls=':',  label = 'shot noise',                   lw=lw, zorder=2 )
                    ax.loglog( dict_Jg_noint["k"].value,  dict_Jg_noint["pk_J"][0].value - dict_J["LIM_shot"][0].value,  c='darkgray', lw=lw,   label = 'non param. $\\rm P^{clust}$(k)',      zorder=3 )
                    ax.errorbar(dict_Jg_noint["k"].value, dict_Jg_noint["pk_J"][0].value - dict_J["LIM_shot"][0].value,     yerr= dict_Jg_noint["pk_J"][1].value, c='darkgray',         lw=lw, zorder=4, elinewidth=lw)
                    ax.loglog( dict_Jg_noint["k"][born[0]:born[1]].value, (dict_Jg_noint["pk_J"][0].value - dict_J["LIM_shot"][0].value)[k_a:k_e+1], 'or',markersize=mk, label='points used in fit', zorder=5)
                    ax.loglog( dict_Jg["k"][born[0]:born[1]], f(dict_Jg["k"][born[0]:born[1]]) * ( bX*I)**2, '--xb',label ='$ (I \\times b^{\\rm fit}_\\mathrm{eff})^2 P^{\\rm 2D}_{\\rm matter}$ \n$ b^{\\rm fit}_\\mathrm{eff}$='+f'{np.round(bX,2)}'+'$\\pm$'+f'{np.round(dbX,2)}', lw=lw,markersize=mk, zorder=6)
                    ax.legend(fontsize=BS-5,loc = 'lower left')
                    """
                    secax = ax.secondary_xaxis("top", functions=(G,G))
                    secax.set_xlabel('k [$\\rm Mpc^{-1}$]')
                    secax = ax.secondary_yaxis("right", functions=(F,F))
                    secax.set_ylabel(f"P{J}(k) "+'[$\\rm Jy^2/sr^2.Mpc^3]$]')
                    """
                    ax=axs[1]
                    ax.set_xlabel("k [$\\rm arcmin^{-1}$]")
                    ax.set_ylabel("$\\rm P_{gal}$(k) [sr]"); #plt.ylim(5e-6,2e-3)
                    ax.set_ylim(4e-9, 2e-5)
                    ax.loglog(  dict_Jg_noint["k"], dict_Jg_noint["pk_gal"][0].value, 'darkgray', ls="--", label = 'non-param. $\\rm P^{tot}$(k)' ,lw=lw, zorder=1)
                    ax.loglog(  dict_Jg["k"],  np.ones(len(dict_Jg["k"]))*dict_J["gal_shot"][0], c='darkgray', ls=':', label = 'shot noise',lw=lw, zorder=2 )
                    ax.loglog(  dict_Jg_noint["k"].value, dict_Jg_noint["pk_gal"][0].value - dict_J["gal_shot"][0],  c='darkgray', lw=lw, label = 'non param. $\\rm P^{clust}$(k)', zorder=3 )
                    ax.errorbar(dict_Jg_noint["k"].value, dict_Jg_noint["pk_gal"][0].value - dict_J["gal_shot"][0], yerr= dict_Jg_noint["pk_gal"][1].value, c='darkgray', lw=lw, zorder=4, elinewidth=lw)
                    ax.loglog(dict_Jg_noint["k"][born[0]:born[1]].value, (dict_Jg_noint["pk_gal"][0].value - dict_J["gal_shot"][0])[k_a:k_e+1], 'or', markersize = mk, label = 'points used in fit' ,zorder=5)
                    ax.loglog(dict_Jg["k"][born[0]:born[1]],  f(dict_Jg["k"][k_a:k_e+1]) * ( bgal)**2, '--xb',  label = '$ ( b^{\\rm fit}_\\mathrm{gal})^2 P^{\\rm 2D}_{\\rm matter}$ \n$ b^{\\rm fit}_\\mathrm{gal}$='+f'{np.round(bgal,2)}'+'$\\pm$'+f'{np.round(dbgal,2)}', lw=lw,markersize=mk, zorder=6)
                    ax.legend(fontsize=BS-5,loc = 'lower left')
                    
                    secax = ax.secondary_xaxis("top", functions=(G,G))
                    secax.set_xlabel('k [$\\rm Mpc^{-1}$]')
                    secax = ax.secondary_yaxis("right", functions=(F,F))
                    secax.set_ylabel("$\\rm P_{gal} $(k) "+'[$\\rm Mpc^3]$')
                    fig.tight_layout()
                    print('save')
                    for extension in ("png", "pdf"): plt.savefig(f"/home/mvancuyck/mathilde_these/plot/example_{J}_z{z}_dz{dz}_n{n_slices}_fit2.{extension}", transparent=True)
                    plt.show()

                    embed()
                    
                #----------------
            
                if(J=='CO76'): 
                    line_noint = f"{simuu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{field_size}deg2_{J}_CI21"
                    dict_Jg = powspec_LIMgal(line, line, galaxy,output_path, 'CI21',  z, dz, n_slices, field_size, dll )
                    Jgal = dict_Jg["pk_J-gal"][0][born[0]:born[1]].value - dict_J["LIMgal_shot"][0].value
                    bI_cI21[nfield, zi] = ( Jgal / bgal / p2d ).mean()
            
    return patchs, bI_mes, bI_cons, bI_cI21, pk_mes, b_mes
    #
  
def b_I_article(dz,n_slices, simu = None, ref = 'CO32',  dtype='all_lines', recompute=False):

    if(simu=='uchuu'): res = res_uchuu;    ksimu = 3.3e-1
    else:              res = res_subfield; ksimu = 3.3e-1 #before 2e-1
    colors = cm.copper(np.linspace(0,1,len(z_list)))
    
    #--- bI from Cross pk ---
    patchs, bI_mes, bI_cons, _, pk_mes, B_mes = bI_from_nsubfields(n_slices, dtype, res, dz, simu,  ksimu, colors, recompute=recompute)
    #Average over subfields
    bI     = np.mean(bI_mes[:,:,:,:,0] ,axis=0); sigma_bI = np.std(bI_mes[:,:,:,:,0],axis=0)
    bIcons = np.mean(bI_cons[:,:,:,:,0],axis=0); sigma_bIcons = np.std(bI_cons[:,:,:,:,0],axis=0)
    
    #No interlopers
    _, bI_noint, _,bI_cI21, pk_mes_noint, _= bI_from_nsubfields(n_slices, None, res, dz, simu,  ksimu, colors, recompute=recompute)
    bI_co = np.mean(bI_noint[:,:,:,:,0] ,axis=0); sigma_bI_co= np.std(bI_noint[:,:,:,:,0],axis=0)
    mean_ci = np.mean(bI_cI21, axis =0); std_ci =  np.std(bI_cI21, axis =0)

    #Only CO interlopers
    _, bI_mes_coall, _,_, _, _= bI_from_nsubfields(n_slices, 'CO_all', res, dz, simu,  ksimu, colors, recompute=recompute)
    bI_coall = np.mean(bI_mes_coall[:,:,:,:,0] ,axis=0); sigma_bI_coall = np.std(bI_mes_coall[:,:,:,:,0],axis=0)
    sum_cross = bI_cI21 + bI_mes_coall[:,:,6,0,0]; mean_cross = np.mean(sum_cross, axis =0); sigma_sum_cross= np.std(sum_cross, axis =0)

    
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(5,5), dpi=200); mks=6; dr = 0.04; dy = 0.0; text_y_offset=0.2; lw=1
    width = [1,1];  height = [1,1,1,1,1]
    drelative = (-4,-3,-2,-1,0,1,2,3)
    gs = gridspec.GridSpec(ncols=2, nrows=5, height_ratios= height, width_ratios=width)
    axr = plt.subplot(gs[-2]); patchs = []
    for j, line, rest_freq in zip(np.arange(8), line_list[:8], rest_freq_list[:8]):
        c=colors_co[j]; cols = cm.rainbow(np.linspace(0,1,12))
        #---------------------------
        ax = plt.subplot(gs[j]); 
        #---------------------------
        if(j==7): patch = mpatches.Patch(color='grey', label=r'from cross-power spectrum estimate' ); patchs.append(patch)
        if(j!=7):
            ax.errorbar(    z_list, bI[:,j,0],  linestyle='solid', color=c, ecolor=c, lw=lw) 
            ax.fill_between(z_list, bI[:,j,0]-sigma_bI[:,j,0], bI[:,j,0]+sigma_bI[:,j,0], alpha = 0.2, color=c)
        else:
            ax.errorbar( z_list[1:], bI[1:,j,0],  linestyle='solid', color=c, ecolor=c, lw=lw) #
            ax.fill_between(z_list[1:], bI[1:,j,0]-sigma_bI[1:,j,0], bI[1:,j,0]+sigma_bI[1:,j,0], alpha = 0.2, color=c)
            ax.errorbar((0.5,),(3e2,), yerr=(10,), uplims=True, marker="None", color=c, markeredgecolor=c, markerfacecolor=c, linestyle="None", alpha=0.5)
            
        if(j==7): patch = mlines.Line2D([], [], color='k', linestyle=":", marker="*",  label='catalogue mean brightness \ntimes effective bias'); patchs.append(patch)
        ax.errorbar( z_list, bIcons[:,j,0], yerr = sigma_bIcons[:,j,0], linestyle=":", color='k', ecolor='k', marker='*',markersize = mks,lw=lw )
        if(j==7): patch = mlines.Line2D([], [], color='k', linestyle="dashdot", marker="p", mfc='none', label=r'$\Sigma (S*b^\mathrm{Tinker+10}_\mathrm{eff})$/$\Omega_\mathrm{field}$'); patchs.append(patch)
        ax.errorbar( z_list, bIcons[:,j,1], yerr = sigma_bIcons[:,j,1], linestyle="dashdot", mfc='none', color='k', ecolor='k', marker='p', markersize = mks, lw=lw )
        axr.errorbar( np.asarray(z_list)+drelative[j]*dr, bIcons[:,j,1]/bI[:,j,0]-1+dy, yerr = sigma_bIcons[:,j,1]/bI[:,j,0], linestyle="None", color=c,  ecolor=c, marker='p', markersize = mks, lw=lw )
        #---------------------------
        ax.set_ylabel(r"$b_{\rm eff} \times$"+f"I {line}")
        if(j==7): ax.legend(handles = patchs, loc='center left', bbox_to_anchor=(-0.3, -0.8), fontsize=7)
        ax.set_xlim(0.4, np.max(z_list)+0.1)
        ax.set_yscale("log")
        if(j<7): ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=False)
        else:
            ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=True)
            ax.set_xlabel(r"redshift")
    axr.set_xlabel(r"redshift")
    axr.plot((0,8), np.zeros(2), c='grey')
    axr.set_ylim(-0.3, 0.2)
    axr.set_ylabel("relative difference"); axr.set_xlim(0.4,  np.max(z_list)+0.1)
    axr.tick_params(axis = "x", which='major', tickdir = "inout", top = True, color='k')
    fig.tight_layout(); fig.subplots_adjust(hspace=.0)
    #---------------------------
    for extension in ("png", "pdf"): plt.savefig(f"bI_CO.{extension}")
    for extension in ("png", "pdf"): plt.savefig(f"{simu}_int{dtype}_{dtype}_bI{line}_vs_z.{extension}")
    plt.show()

    
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(3,3), dpi=200); lw=1; mks=6; j=6; c=colors_co[6]
    plt.errorbar(    z_list, bI[:,j,0], linestyle='solid', color=c, ecolor=c, label="CO cross-power,\n with interlopers", lw=lw) 
    plt.fill_between(z_list, bI[:,j,0]-sigma_bI[:,j,0], bI[:,j,0]+sigma_bI[:,j,0], alpha = 0.2, color=c)
    plt.errorbar(    z_list, bIcons[:,j,0], yerr = sigma_bIcons[:,j,0], linestyle=":", color='k', ecolor='k', marker='*',markersize = mks, label = 'catalogue mean brightness \ntimes fitted effective bias', lw=lw)
    c='grey'
    plt.errorbar(    z_list, mean_ci, linestyle='--', color=c, ecolor=c,  label='[CI]21 cross-power', lw=lw) 
    plt.fill_between(z_list, mean_ci - std_ci, mean_ci +  std_ci, alpha = 0.2, color=c,)
    c='purple'
    plt.errorbar( z_list, bI_coall[:,j,0],  linestyle='--', color=c, ecolor=c,  label='CO76 cross-power', lw=lw) # yerr = sigma_bI_all_subfiels[:,j,0],
    plt.fill_between(z_list, bI_coall[:,j,0] -  sigma_bI_coall[:,j,0], bI_coall[:,j,0] +  sigma_bI_coall[:,j,0], alpha = 0.2, color=c,)
    plt.errorbar(z_list, mean_cross, yerr = sigma_sum_cross, ls='--',c='dimgray',lw=lw, label='(CO76+[CI]21) cross-power)')    
    plt.ylabel(r"$b_{\rm eff} \times$"+f"I CO76",)
    plt.legend(loc='lower right', fontsize=BS-2)
    plt.xlim(0.4, np.max(z_list)+0.1); plt.ylim(5e1,8e2)
    plt.yscale("log")
    plt.xlabel(r"redshift")
    plt.tight_layout()
    for extension in ("png", "pdf"): plt.savefig(f"CO76.{extension}")



    
  
def b_I(dz,n_slices, simu = None, ref = 'CO32',  dtype='all_lines', recompute=False):

    if(simu=='uchuu'): res = res_uchuu;    ksimu = 3.3e-1
    else:              res = res_subfield; ksimu = 3.3e-1 #before 2e-1
    colors = cm.copper(np.linspace(0,1,len(z_list)))
    
    #--- bI from Cross pk ---
    patchs, bI_mes, bI_cons, _, pk_mes, B_mes = bI_from_nsubfields(n_slices, dtype, res, dz, simu,  ksimu, colors, recompute=recompute)
    #Average over subfields
    bI     = np.mean(bI_mes[:,:,:,:,0] ,axis=0); sigma_bI = np.std(bI_mes[:,:,:,:,0],axis=0)
    bIcons = np.mean(bI_cons[:,:,:,:,0],axis=0); sigma_bIcons = np.std(bI_cons[:,:,:,:,0],axis=0)
    
    #No interlopers
    _, bI_noint, _,bI_cI21, pk_mes_noint, _= bI_from_nsubfields(n_slices, None, res, dz, simu,  ksimu, colors, recompute=recompute)
    bI_co = np.mean(bI_noint[:,:,:,:,0] ,axis=0); sigma_bI_co= np.std(bI_noint[:,:,:,:,0],axis=0)
    mean_ci = np.mean(bI_cI21, axis =0); std_ci =  np.std(bI_cI21, axis =0)

    #Only CO interlopers
    _, bI_mes_coall, _,_, _, _= bI_from_nsubfields(n_slices, 'CO_all', res, dz, simu,  ksimu, colors, recompute=recompute)
    bI_coall = np.mean(bI_mes_coall[:,:,:,:,0] ,axis=0); sigma_bI_coall = np.std(bI_mes_coall[:,:,:,:,0],axis=0)
    sum_cross = bI_cI21 + bI_mes_coall[:,:,6,0,0]; mean_cross = np.mean(sum_cross, axis =0); sigma_sum_cross= np.std(sum_cross, axis =0)

    embed()
    patchs=[]
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(5,5), dpi=200); mks=6; dr = 0.04; dy = 0.0; text_y_offset=0.2; lw=1
    width = [1,1];  height = [1,1,1,1]
    drelative = (-4,-3,-2,-1,0,1,2,3)
    gs = gridspec.GridSpec(ncols=2, nrows=4, height_ratios= height, width_ratios=width)
    #axr = plt.subplot(gs[-2]); patchs = []
    for j, line, rest_freq in zip(np.arange(6), line_list[:6], rest_freq_list[:6]):
        c=colors_co[j]; cols = cm.rainbow(np.linspace(0,1,12))
        #---------------------------
        ax = plt.subplot(gs[j]); 
        #---------------------------
        if(j==5): patch = mpatches.Patch(color='grey', label=r'from cross-power spectrum estimate' ); patchs.append(patch)
        ax.errorbar(    z_list, bI[:,j,0],  linestyle='solid', color=c, ecolor=c, lw=lw) 
        ax.fill_between(z_list, bI[:,j,0]-sigma_bI[:,j,0], bI[:,j,0]+sigma_bI[:,j,0], alpha = 0.2, color=c)
        if(j==5): patch = mlines.Line2D([], [], color='k', linestyle=":", marker="*",  label='catalogue mean brightness \ntimes effective bias'); patchs.append(patch)
        #ax.errorbar( z_list, bIcons[:,j,0], yerr = sigma_bIcons[:,j,0], linestyle=":", color='k', ecolor='k', marker='*',markersize = mks,lw=lw )
        if(j==5): patch = mlines.Line2D([], [], color='k', linestyle="dashdot", marker="p", mfc='none', label=r'Tinker et al. 2010'); patchs.append(patch)
        #ax.errorbar( z_list, bIcons[:,j,1], yerr = sigma_bIcons[:,j,1], linestyle="dashdot", mfc='none', color='k', ecolor='k', marker='p', markersize = mks, lw=lw )
        if(j==5): ax.legend(handles = patchs, loc='center left', bbox_to_anchor=(-0.3, -0.6), fontsize=7)
        #axr.errorbar( np.asarray(z_list)+drelative[j]*dr, bIcons[:,j,1]/bI[:,j,0]-1+dy, yerr = sigma_bIcons[:,j,1]/bI[:,j,0], linestyle="None", color=c,  ecolor=c, marker='p', markersize = mks, lw=lw )
        print(f'Mean uncertainty of J={j+1}:'+f'{100*np.mean( sigma_bI[:,j,0]/bI[:,j,0] )}'+'%')
        print(f'Mean diff with intrinsec of J={j+1}:'+f'{100*np.mean( bI[:,j,0]/bIcons[:,j,0]-1)}'+'%')
        print('')
        #---------------------------
        ax.set_ylabel(r"$b_{\rm eff} \times$"+f"I {line}")
        ax.set_xlim(0.4, np.max(z_list)+0.1)
        ax.set_yscale("log")
        if(j<5): ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=False)
        else:
            ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=True)
            ax.set_xlabel(r"redshift")
    fig.tight_layout(); fig.subplots_adjust(hspace=.0)
    
    #---------------------------
    for extension in ("png", "pdf"): plt.savefig(f"bI_CO.{extension}")
    for extension in ("png", "pdf"): plt.savefig(f"{simu}_int{dtype}_{dtype}_bI{line}_vs_z.{extension}")
    plt.show()

    embed()
    
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(3,3), dpi=200); lw=1; mks=6; j=6; c=colors_co[6]
    plt.errorbar(    z_list, bI[:,j,0], linestyle='solid', color=c, ecolor=c, label="CO cross-power,\n with interlopers", lw=lw) 
    plt.fill_between(z_list, bI[:,j,0]-sigma_bI[:,j,0], bI[:,j,0]+sigma_bI[:,j,0], alpha = 0.2, color=c)
    plt.errorbar(    z_list, bIcons[:,j,0], yerr = sigma_bIcons[:,j,0], linestyle=":", color='k', ecolor='k', marker='*',markersize = mks, label = 'catalogue mean brightness \ntimes fitted effective bias', lw=lw)
    c='grey'
    plt.errorbar(    z_list, mean_ci, linestyle='--', color=c, ecolor=c,  label='[CI]21 cross-power', lw=lw) 
    plt.fill_between(z_list, mean_ci - std_ci, mean_ci +  std_ci, alpha = 0.2, color=c,)
    c='purple'
    plt.errorbar( z_list, bI_coall[:,j,0],  linestyle='--', color=c, ecolor=c,  label='CO76 cross-power', lw=lw) # yerr = sigma_bI_all_subfiels[:,j,0],
    plt.fill_between(z_list, bI_coall[:,j,0] -  sigma_bI_coall[:,j,0], bI_coall[:,j,0] +  sigma_bI_coall[:,j,0], alpha = 0.2, color=c,)
    plt.errorbar(z_list, mean_cross, yerr = sigma_sum_cross, ls='--',c='dimgray',lw=lw, label='(CO76+[CI]21) cross-power)')    
    plt.ylabel(r"$b_{\rm eff} \times$"+f"I CO76",)
    plt.legend(loc='lower right', fontsize=BS-2)
    plt.xlim(0.4, np.max(z_list)+0.1); plt.ylim(5e1,8e2)
    plt.yscale("log")
    plt.xlabel(r"redshift")
    plt.tight_layout()
    for extension in ("png", "pdf"): plt.savefig(f"CO76.{extension}") 


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="co paper main",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    #options
    parser.add_argument('--recompute', help = "recompute cross power spectra", action="store_true")

    args = parser.parse_args()
    
    b_I(dz=0.05, n_slices=2.0, dtype='all_lines', recompute=args.recompute)
    
