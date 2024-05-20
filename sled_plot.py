from fcts import *
from gen_all_sizes_cubes import *
from functools import partial
from multiprocessing import Pool, cpu_count
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
import matplotlib.cm as cm
from progress.bar import Bar
import argparse
from pysides.load_params import *
 

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





def compute_sled_cat(z_list, dz_list, simu = 'uchuu', recompute = False):
    #choose the type of frequency interval: fixe for one transition of fix redshift interval for all transitions
    if(recompute):

        if(simu == 'bolshoi'): 
            #With SIDES Bolshoi, for rapid tests. 
            dirpath="/home/mvancuyck/"
            cat = Table.read(dirpath+'pySIDES_from_original.fits')
            cat = cat.to_pandas(); simu = 'bolshoi'
        else: simu, cat, cat_path, fs = load_cat()

        bar = Bar('computing SLED from catalog', max=len(z_list)*len(dz_list)*8*3)  
        for type in ('all', 'ms', 'sb'):

            name_I = f'dict_dir/I_mean_of_uchuu_n0_{type}.npy'
            #Compute the mean brighness of CO transitions up to J=8 with different ways for each redshift interval
            I_dict = np.zeros((len(z_list), len(dz_list), 8,16))

            for iz, z in enumerate(z_list):
                for idz, dz in enumerate(dz_list):


                    if(simu != 'bolshoi'):
                        if(type=='ms'): type_cat = cat.loc[cat["issb"] ==70]
                        if(type=='sb'): type_cat = cat.loc[cat["issb"] !=70]
                        else: type_cat = cat
                    else: 
                        if(type=='ms'): type_cat = cat.loc[cat["issb"] ==False]
                        if(type=='sb'): type_cat = cat.loc[cat["issb"] !=True]
                        else: type_cat = cat

                    for j, (line, rest_freq) in enumerate(zip(line_list[:8], rest_freq_list[:8])):
                        
                        #compute rest and observed frequencies at redshift z
                        rest_freq = rest_freq.value
                        nu_obs = rest_freq / (1+z)
                            
                        #Set the frequency intervall according to the choosen receipe
                        dnu = nu_obs* dz/(1+z)
                            
                        #compute the size of the associate frequency slice
                        nu_min = nu_obs - (0*dnu) 
                        nu_max = nu_obs + (0*dnu) 
                        nu_min_edge = nu_min - dnu/2
                        nu_max_edge = nu_max + dnu/2
                            
                        #select sources within the frequency slice at redshift of interest.
                        cat_line = type_cat.loc[ np.abs(  rest_freq/(1+type_cat['redshift'])  - nu_obs) <= (nu_max_edge - nu_min_edge)/2]
                            
                        #Mean brightness of the line
                        I_dict[iz, idz, j,0] = cat_line[f"I{line}"].mean() #Jy.km/s
                        I_dict[iz, idz, j,1] = cat_line[f"I{line}"].std()
                        I_dict[iz, idz, j,2] = cat_line[f"I{line}"].median()
                        I_dict[iz, idz, j,3] = np.quantile(cat_line[f"I{line}"], 0.25)
                        I_dict[iz, idz, j,4] = np.quantile(cat_line[f"I{line}"], 0.75)
                        I_dict[iz, idz, j,5] = cat_line[f"I{line}"].sum()
                        #brightness ratio wrt CO32
                        I_dict[iz, idz, j,6] = (cat_line[f"I{line}"]/cat_line[f"ICO32"]).mean() #Jy.km/s
                        I_dict[iz, idz, j,7] = (cat_line[f"I{line}"]/cat_line[f"ICO32"]).std()
                        I_dict[iz, idz, j,8] = (cat_line[f"I{line}"]/cat_line[f"ICO32"]).median()
                        I_dict[iz, idz, j,9] = np.quantile(cat_line[f"I{line}"]/cat_line[f"ICO32"],0.25)
                        I_dict[iz, idz, j,10] = np.quantile(cat_line[f"I{line}"]/cat_line[f"ICO32"],0.75)
                        #brightness ratio wrt CO10
                        I_dict[iz, idz, j,11] = (cat_line[f"I{line}"]/cat_line[f"ICO10"]).mean() #Jy.km/s
                        I_dict[iz, idz, j,12] = (cat_line[f"I{line}"]/cat_line[f"ICO10"]).std()
                        I_dict[iz, idz, j,13] = (cat_line[f"I{line}"]/cat_line[f"ICO10"]).median() #Jy.km/s
                        I_dict[iz, idz, j,14] =  np.quantile(cat_line[f"I{line}"]/cat_line[f"ICO10"],0.25)
                        I_dict[iz, idz, j,15] =  np.quantile(cat_line[f"I{line}"]/cat_line[f"ICO10"],0.75)
                        bar.next() 

            np.save(name_I, np.asarray(I_dict), 'wb')  

        bar.finish

    all_I = np.load(f'dict_dir/I_mean_of_uchuu_n0_all.npy')
    ms_I  = np.load(f'dict_dir/I_mean_of_uchuu_n0_ms.npy')
    sb_I  = np.load(f'dict_dir/I_mean_of_uchuu_n0_sb.npy')

    return all_I, ms_I, sb_I


def plot_sled_fig(z_list, dz_list, recompute_sleds):
    '''
    Change the ref CO transition by hand!!!
    '''

    I_dict, ms_I, sb_I = compute_sled_cat(z_list, dz_list, simu = 'bolshoi', recompute = recompute_sleds)

    for idz, dz in enumerate(dz_list):

        BS=10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); lw=1; mk=5; elw=1
        fig = plt.figure(figsize=(4,4), dpi=200) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]); ax = plt.subplot(gs[0]);  axr = plt.subplot(gs[1])

        patchs = []
        patch = mlines.Line2D([], [], color='k', linestyle='solid',  label='Mean SLED')
        patchs.append(patch)
        patch = mlines.Line2D([], [], color='k', linestyle='None', marker='*', label=f'from cross-power spectra')
        patchs.append(patch)

        #--- SLED from Catalog ---
        for zi, (z, c) in enumerate(zip(z_list, cm.copper(np.linspace(0,0.8,len(z_list))) )):
            J_list = np.arange(I_dict.shape[2])+1
            N=6; M=N+1
            ax.errorbar(     J_list, I_dict[zi, idz, :,N], color=c, ecolor=c, lw=1)
            
            ax.fill_between( J_list, I_dict[zi, idz, :,N] - I_dict[zi, idz, :, M], 
                                     I_dict[zi, idz, :,N] + I_dict[zi, idz, :, M], 
                                     color=c, alpha=0.3 )
        #---------------------------

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
        for extension in ("png", "pdf"): plt.savefig(f"figs/sled_dz{dz}.{extension}", transparent=True)


def contrib_ms_sb(z_list, dz_list, recompute_sleds): 

    all_I, ms_I, sb_I = compute_sled_cat(z_list, dz_list, simu = 'bolshoi', recompute = recompute_sleds)

    for idz, dz in enumerate(dz_list):
        #---------------------------
        BS=10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); mks=6; lw=2
        fig = plt.figure(figsize=(6,3), dpi=200) 
        #for z, zi, c in zip(z_list, range(len(z_list)), colors ):
        patchs = []
        colors_co = ('orange', 'r', 'b', 'cyan', 'g', 'purple', 'magenta', 'grey',)
         
        for j, (line, rest_freq, c) in enumerate(zip(line_list[:8], rest_freq_list[:8], colors_co)):
            plt.errorbar( z_list, 100 -100 * (sb_I[:, idz, j, 5] / all_I[:, idz, j, 5]), c=c, fmt='--D', lw=lw, markersize=mks)
            patch = mlines.Line2D([], [], color=c, linestyle="--", marker="D", label='$\\rm J_{up}$'+f'={j+1}', markersize=mks, lw=lw); patchs.append(patch)
        
        plt.legend(handles = patchs, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=BS)
        plt.xlabel('redshift')
        plt.yscale("log")
        plt.ylabel('Contribution of SB objects to $\\rm I_{J_{up}}$ [%]')
        plt.tight_layout()
        for extension in ("png", "pdf"): plt.savefig(f"figs/sb_ms.{extension}", transparent=True)
        
    plt.show()

    
if __name__ == "__main__":

    #python SLIM_powspec_species.py --recompute to recompute all pks
    parser = argparse.ArgumentParser(description="recompute Uchuu SLED",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--recompute_sleds', help = "recompute CO sled from Uchuu catalogue", action="store_true")
    args = parser.parse_args()

    #With SIDES Bolshoi, for rapid tests. 
    tim_params = load_params('PAR/cubes.par')
    z_list = tim_params['z_list']
    dz_list = tim_params['dz_list']
    n_list = tim_params['n_list']

    if(False):
        plot_sled_fig(z_list, dz_list, args.recompute_sleds)
        plt.show()

    contrib_ms_sb(z_list, dz_list, args.recompute_sleds)
