from gen_cube_co_paper import *
from fct_co_paper import *
from functools import partial
from multiprocessing import Pool, cpu_count
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
import matplotlib.cm as cm
from progress.bar import Bar
import argparse

def compute_dA(A,B,dB,C,dC, includenewaxis = False):
    if(includenewaxis): return np.sqrt((dB/B)**2+( (dC/C)[:,:,np.newaxis] )**2)*A
    else: return np.sqrt((dB/B)**2+(dC/C)**2)*A

    
def study_contrib_to_sled(dz):

    all_I = np.load(f'I_mean_of_uchuu_dz{dz}_n2.0_all.npy')
    ms_I  = np.load(f'I_mean_of_uchuu_dz{dz}_n2.0_ms.npy')
    sb_I  = np.load(f'I_mean_of_uchuu_dz{dz}_n2.0_sb.npy')

    #---------------------------
    BS=10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); mks=6; lw=2
    fig = plt.figure(figsize=(6,3), dpi=200) 
    #for z, zi, c in zip(z_list, range(len(z_list)), colors ):
    patchs = []
    for j, line, rest_freq in zip(np.arange(8), line_list[:8], rest_freq_list[:8]):
        c=colors_co[j];
        plt.errorbar( z_list, 100* sb_I[:, j, 5] / all_I[:, j, 5], c=c, fmt='--D', lw=lw, markersize=mks)
        patch = mlines.Line2D([], [], color=c, linestyle="--", marker="D", label=f'J={j+1}', markersize=mks, lw=lw); patchs.append(patch)
    plt.legend(handles = patchs, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=BS)
    plt.xlabel('redshift')
    plt.yscale("log")
    plt.ylabel('Contribution of SB objects to $\\rm I_J$ [%]')
    plt.tight_layout()
    for extension in ("png", "pdf"): plt.savefig(f"sb_ms_2.{extension}")

            
def mean_SLED_uchuu(dz, n_avg=2, recompute = False):
    #choose the type of frequency interval: fixe for one transition of fix redshift interval for all transitions
    for type in ('all', 'ms', 'sb'):
        name_I = f'I_mean_of_uchuu_dz{dz}_n{n_avg}_{type}.npy'
        if(not os.path.isfile(name_I) or recompute):
            embed()
            simu, cat, cat_path, fs = load_cat()
        else: cat=None
    #For all objects then MS objects and SB objects: 
    for type in ('all', 'ms', 'sb'):
        name_I = f'I_mean_of_uchuu_dz{dz}_n{n_avg}_{type}.npy'
        if(not os.path.isfile(name_I) or recompute):
            if(type=='ms'): type_cat = cat.loc[cat["issb"] ==70]
            if(type=='sb'): type_cat = cat.loc[cat["issb"] !=70]
            else: type_cat = cat
            #Compute the mean brighness of CO transitions up to J=8 with different ways for each redshift interval
            I_dict = np.zeros((len(z_list), 8,16))
            bar = Bar('Processing', max=len(z_list)*8)              
            for z, Z in zip(z_list, range(len(z_list))):
                for j, line, rest_freq in zip(np.arange(8), line_list[:8], rest_freq_list[:8]):
                    #compute rest and observed frequencies at redshift z
                    rest_freq = rest_freq.value
                    nu_obs = rest_freq / (1+z)
                    #Set the frequency intervall according to the choosen receipe
                    dnu = nu_obs* dz/(1+z)
                    #compute the size of the associate frequency slice
                    nu_min = nu_obs - (n_avg*dnu) 
                    nu_max = nu_obs + (n_avg*dnu) 
                    nu_min_edge = nu_min - dnu/2
                    nu_max_edge = nu_max + dnu/2
                    #select sources within the frequency slice at redshift of interest.
                    cat_line = type_cat.loc[ np.abs(  rest_freq/(1+type_cat['redshift'])  - nu_obs) <= (nu_max_edge - nu_min_edge)/2]
                    #Mean brightness of the line
                    I_dict[Z, j,0] = cat_line[f"I{line}"].mean() #Jy.km/s
                    I_dict[Z, j,1] = cat_line[f"I{line}"].std()
                    I_dict[Z, j,2] = cat_line[f"I{line}"].median()
                    I_dict[Z, j,3] = np.quantile(cat_line[f"I{line}"], 0.25)
                    I_dict[Z, j,4] = np.quantile(cat_line[f"I{line}"], 0.75)
                    I_dict[Z, j,5] = cat_line[f"I{line}"].sum()
                    #brightness ratio wrt CO32
                    I_dict[Z, j,6] = (cat_line[f"I{line}"]/cat_line[f"ICO32"]).mean() #Jy.km/s
                    I_dict[Z, j,7] = (cat_line[f"I{line}"]/cat_line[f"ICO32"]).std()
                    I_dict[Z, j,8] = (cat_line[f"I{line}"]/cat_line[f"ICO32"]).median()
                    I_dict[Z, j,9] = np.quantile(cat_line[f"I{line}"]/cat_line[f"ICO32"],0.25)
                    I_dict[Z, j,10] = np.quantile(cat_line[f"I{line}"]/cat_line[f"ICO32"],0.75)
                    #brightness ratio wrt CO10
                    I_dict[Z, j,11] = (cat_line[f"I{line}"]/cat_line[f"ICO10"]).mean() #Jy.km/s
                    I_dict[Z, j,12] = (cat_line[f"I{line}"]/cat_line[f"ICO10"]).std()
                    I_dict[Z, j,13] = (cat_line[f"I{line}"]/cat_line[f"ICO10"]).median() #Jy.km/s
                    I_dict[Z, j,14] =  np.quantile(cat_line[f"I{line}"]/cat_line[f"ICO10"],0.25)
                    I_dict[Z, j,15] =  np.quantile(cat_line[f"I{line}"]/cat_line[f"ICO10"],0.75)
                    bar.next()
            bar.finish
            np.save(name_I, np.asarray(I_dict), 'wb')        
    all_I = np.load(f'I_mean_of_uchuu_dz{dz}_n{n_avg}_all.npy')
    ms_I  = np.load(f'I_mean_of_uchuu_dz{dz}_n{n_avg}_ms.npy')
    sb_I  = np.load(f'I_mean_of_uchuu_dz{dz}_n{n_avg}_sb.npy')

    return all_I, ms_I, sb_I

def co_sled_from_nsubfields(i_ref,n_slices,dz, res, field_size, ksimu,colors, interlopers = None, allpoints=False):
    #---------------------------
    SLED_mes = np.zeros((12, len(z_list), 9, 2)) #12 subfields, redshift, ref+8lines, mean and std
    patchs = []
    #for each redshift and subfield: 
    for z, zi, c in zip(z_list, range(len(z_list)), colors ):
        for nfield in range(12):
            #Load the name of the subfield
            simuu, cat, cat_path, field_size = load_9deg_subfield(nfield, load_cat = False)
            #-----------------
            #Load the dicts for transition of reference 
            line_noint = f"{simuu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{field_size}deg2_{line_list[i_ref]}"
            galaxy = f"{simuu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{field_size}deg2_galaxies_{line_list[i_ref]}"
            dict_Jg_noint = powspec_LIMgal(                    line_noint, line_noint, galaxy,output_path, line_list[i_ref],  z, dz, n_slices, field_size, dll)
            dict_J        = compute_other_linear_model_params( line_noint, line_noint, output_path, line_list[i_ref], rest_freq_list[i_ref], z, dz, n_slices, field_size, cat, dict_Jg_noint)
            #-----------------                
            #Without interlopers
            if(interlopers == None): line =  line_noint; dict_Jg = dict_Jg_noint 
            else: line = line_noint + f'_{interlopers}'; dict_Jg = powspec_LIMgal(line, line, galaxy,output_path,  line_list[i_ref],  z, dz, n_slices, field_size, dll)
            #-----------------
            #Get the scales
            k_matter_3D = (dict_Jg["k"].to(u.rad**-1) * 2 * np.pi  /  dict_J["Dc"]).value
            w = np.where( k_matter_3D <= ksimu)
            k_a = w[0][0]; k_e =  w[0][-1]
            if(k_e <= 1):  k_e = 2
            if(allpoints): born = [0,-1]
            else: born = [int(k_a),int(k_e+1)]
            intK, slopeK = print_scientific_notation( np.mean(k_matter_3D[k_e]) )
            #create patchs for legend
            if(nfield ==0):
                patch = mpatches.Patch(color=c, label=f'z={z}')#+"@k$\\leq$"+f"({intK}"+r"$\times$"+r"$10^{-1}$)"+r"$ \rm Mpc^{-1}$")
                patchs.append(patch)
            #Get the power spectrum's mean and dispersion for the line of reference
            SLED_mes[nfield, zi, 0, 0] = (dict_Jg['pk_J-gal'][0][born[0]:born[1]].value - dict_J["LIMgal_shot"][0].value).mean()
            SLED_mes[nfield, zi, 0, 1] = (dict_Jg['pk_J-gal'][0][born[0]:born[1]].value - dict_J["LIMgal_shot"][0].value).std()
            pk_ref                     = (dict_Jg['pk_J-gal'][0][born[0]:born[1]].value - dict_J["LIMgal_shot"][0].value).mean()
            #----------------
            #Get the power spectrum's mean and dispersion for the line of reference
            for j, J, rest_freq in zip(np.arange(8), line_list[:8], rest_freq_list[:8]):
                line_noint = f"{simuu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{field_size}deg2_{J}"
                galaxy = f"{simuu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{field_size}deg2_galaxies_{J}"
                dict_J = compute_other_linear_model_params( line_noint, line_noint, output_path, J, rest_freq, z, dz, n_slices, field_size, cat, dict_Jg_noint)
                if(interlopers == None): line = line_noint
                else:                    line = line_noint + f'_{interlopers}'
                dict_Jg = powspec_LIMgal(line, line, galaxy,output_path, J,  z, dz, n_slices, field_size, dll)
                SLED_mes[nfield, zi, j+1, 0] = (dict_Jg['pk_J-gal'][0][born[0]:born[1]].value - dict_J["LIMgal_shot"][0].value).mean()
                SLED_mes[nfield, zi, j+1, 1] = (dict_Jg['pk_J-gal'][0][born[0]:born[1]].value - dict_J["LIMgal_shot"][0].value).std()
                
    return patchs, SLED_mes

def compute_sled(recompute_sleds = False, ref = 'CO32', n_avg=2.0, dz=0.05,interlopers='all_lines'):

    #Set simu params
    field_size = 9*u.deg**2; res = res_subfield; ksimu = 3.3e-1
    #indice of the transition of reference in line_list
    i_ref = int(ref[-1])
    #Quantum rotational number list
    Jrot_list = np.arange(8)+1
    #redshifts color code
    colors = cm.copper(np.linspace(0,0.8,len(z_list)))
    #--- SLED from Cross pk ---
    patchs, SLED_mes = co_sled_from_nsubfields(i_ref,n_avg,dz, res,field_size,ksimu, colors, interlopers)   
    #Normalisation by J_ref
    B = SLED_mes[:,:,1:,0]
    C = SLED_mes[:,:,0, 0]
    A = B/C[:,:,np.newaxis]
    SLED_mean   = np.mean(A, axis=0)
    SLED_1sigma = np.std( A, axis=0)
    SLED_median = np.median( A, axis=0)
    SLED_1stq = np.quantile( A, 0.25, axis=0)
    SLED_3rdq = np.quantile( A, 0.75, axis=0)
    
    #--- SLED from Catalog ---
    I_dict, I_ms, I_sb = mean_SLED_uchuu(dz, n_avg, recompute=recompute_sleds)
    #--------------------------
    #patch for legend
    #Figure' parameters
    #---------------------------
    
    patch = mlines.Line2D([], [], color='k', linestyle='--',  label='Mean SLED')
    patchs.append(patch)
    patch = mlines.Line2D([], [], color='k', linestyle='None', marker='*', label=f'from cross-power spectra')
    patchs.append(patch)
    
    BS=10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); lw=1; mk=5; elw=1
    fig = plt.figure(figsize=(4,4), dpi=200) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]); ax = plt.subplot(gs[0]);  axr = plt.subplot(gs[1])
    #---------------------------
    for z, zi, c in zip(z_list, range(len(z_list)), colors ):
        N=6; M=N+1
        ax.errorbar( Jrot_list,I_dict[zi,:,N],fmt = '--', color=c, ecolor=c, lw=1)
        ax.fill_between( Jrot_list,I_dict[zi,:,N] - I_dict[zi,:,M], I_dict[zi,:,N] + I_dict[zi,:,M], color=c, alpha=0.3  )
        ax.errorbar( Jrot_list-0.2+0.1*z, SLED_mean[zi,:], yerr=SLED_1sigma[zi,:],  fmt = '*',color=c, ecolor=c, markersize=mk, elinewidth=elw)
        axr.errorbar(Jrot_list-0.2+0.1*z, SLED_mean[zi,:]/I_dict[zi,:,N]-1, yerr =SLED_1sigma[zi,:]/I_dict[zi,:,N] , fmt = '*',color=c, ecolor=c, markersize=mk, elinewidth=elw)

    for J in range(1,7):
        print(f'{J+1}: {np.mean(100 * (I_dict[0,J,M]/I_dict[0,J,N]) )}')
    ax.set_ylabel(r"$ \rm R_\mathrm{J-3} = I_{CO(J-J-1)}/I_{CO(3-2)} $")
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(0.2, 2.5)
    ax.legend(handles = patchs, loc= 'upper left',fontsize=5) #4.5
    #---------------------------
    ax = plt.subplot(gs[1])
    ax.plot((0,9), np.zeros(2), c='grey', lw=lw)
    ax.set_ylabel("relative difference")
    ax.set_xlabel(r"Quantum rotational number J")
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(-0.2,1.6)        
    ax.tick_params(axis = "x", which='major', tickdir = "inout", top = True, color='k')
    #---------------------------
    fig.tight_layout(); fig.subplots_adjust(hspace=.0)
    for extension in ("png", "pdf"): plt.savefig(f"sled.{extension}")
    plt.show()

    embed()
    
if __name__ == "__main__":

    #python SLIM_powspec_species.py --recompute to recompute all pks
    parser = argparse.ArgumentParser(description="recompute Uchuu SLED",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--recompute_sleds', help = "recompute CO sled from Uchuu catalogue", action="store_true")
    args = parser.parse_args()

    compute_sled(recompute_sleds = args.recompute_sleds)
    
