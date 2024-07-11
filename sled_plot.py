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
            I_dict = np.zeros((len(z_list), len(dz_list), 8,17))

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

                        I_dict[iz, idz, j,16] =  len(cat_line)

                        bar.next() 

            np.save(name_I, np.asarray(I_dict), 'wb')  

        bar.finish

    all_I = np.load(f'dict_dir/I_mean_of_uchuu_n0_all.npy')
    ms_I  = np.load(f'dict_dir/I_mean_of_uchuu_n0_ms.npy')
    sb_I  = np.load(f'dict_dir/I_mean_of_uchuu_n0_sb.npy')

    return all_I, ms_I, sb_I

def plot_sled_fig(nslice, z_list, dz_list, recompute_sleds, toembed=False, dtype='_with_interlopers'): #_with_interlopers

    '''
    Change the ref CO transition by hand!!!
    '''


    I_dict, ms_I, sb_I = compute_sled_cat(z_list, dz_list, simu = 'uchuu', recompute = recompute_sleds)
    SLED_mes = co_sled_from_nsubfields(nslice, 2, z_list, dz_list, 9, 0.15, dtype=dtype, toembed=toembed)

    dict = {'SLED_mes':SLED_mes}
    pickle.dump(dict, open(f'dict_dir/SLED_mes_9deg2_dtype{dtype}.p', 'wb'))
    print(line_list)
    for idz, dz in enumerate(dz_list):

        BS=10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); lw=1; mk=5; elw=1
        fig, (ax, axr) = plt.subplots(2, 1, sharex=True, sharey = 'row', gridspec_kw={'height_ratios': [2,1]}, figsize=(4,4), dpi = 200)

        patchs = []
        patch = mlines.Line2D([], [], color='k', linestyle='solid',  label='Mean SLED')
        patchs.append(patch)
        patch = mlines.Line2D([], [], color='k', linestyle='None', marker='o', label=f'from cross-power \n spectra')
        patchs.append(patch)

        #--- SLED from Catalog ---
        
        for zi, (z, c, shift) in enumerate(zip(z_list,  cm.viridis(np.linspace(0.,0.8,len(z_list))),  (-0.2, -0.1, 0, 0.1, 0.2, 0.3))): #(-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4),
            J_list = np.arange(I_dict.shape[2])+1
            N=6; M=N+1

            ax.errorbar(J_list+shift, np.mean(SLED_mes[:, zi, idz, 1:], axis=0), yerr=np.std(SLED_mes[:, zi, idz, 1:], axis=0), 
                        fmt='o', color=c, ecolor=c, lw=1, markersize=2)


            ax.errorbar(     J_list+shift, I_dict[zi, idz, :,N], color=c, ecolor=c, lw=1)
            ax.fill_between( J_list+shift, I_dict[zi, idz, :,N] - I_dict[zi, idz, :, M], 
                                     I_dict[zi, idz, :,N] + I_dict[zi, idz, :, M], 
                                     color=c, alpha=0.2 )
            
            axr.errorbar(J_list+shift, np.mean(SLED_mes[:, zi, idz, 1:], axis=0) / I_dict[zi, idz, :,N]-1, 
                         yerr=np.std(SLED_mes[:, zi, idz, 1:], axis=0) / I_dict[zi, idz, :,N], fmt='o', c=c, markersize=2, lw=1)
            print('z=',z,',', np.round(np.std(SLED_mes[:, zi, idz, 1:], axis=0) / I_dict[zi, idz, :,N], 2),)

            patch = mpatches.Patch(color=c, label=f'z={z}')
            patchs.append(patch)
        #---------------------------
        ax.set_ylabel(r"$ \rm R_\mathrm{J_{up}-3} = B^{CO(J_{up}-J_{up}-1)}/B^{CO(3-2)} $")
        ax.legend(handles = patchs, loc= 'upper left',fontsize=6, frameon = False) #4.5
        ax.set_ylim(0,3)
        #---------------------------
        axr.set_ylim(-0.5,1)
        if(nslice==2.0): axr.set_ylim(-0.4,0.4)

        axr.plot((0,9), np.zeros(2), c='grey', lw=lw)
        axr.set_ylabel("relative \n difference")
        axr.set_xlabel("Quantum rotational number $\\rm J_{up}$")
        axr.set_xlim(0.5, 8.5)
        #axr.tick_params(axis = "x", which='major', tickdir = "inout", top = True, color='k')
        fig.tight_layout(); fig.subplots_adjust(hspace=.0)
        for extension in ("png", "pdf"): plt.savefig(f"sled_dz{dz}_nslice{nslice}.{extension}", transparent=True)

        #plot without catalogue lines and shaded areas. 
        '''
        BS=10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); lw=1; mk=5; elw=1
        fig, (ax, axr) = plt.subplots(2, 1, sharex=True, sharey = 'row', gridspec_kw={'height_ratios': [2,1]}, figsize=(4,4), dpi = 200)
        patchs = []
        patch = mlines.Line2D([], [], color='k', linestyle='solid',  label='Mean SLED')
        patchs.append(patch)
        patch = mlines.Line2D([], [], color='k', linestyle='None', marker='o', label=f'from cross-power \n spectra')
        patchs.append(patch)       
        for zi, (z, c, shift) in enumerate(zip(z_list,  cm.viridis(np.linspace(0.,0.8,len(z_list))),  (-0.2, -0.1, 0, 0.1, 0.2, 0.3))): #(-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4),
            J_list = np.arange(I_dict.shape[2])+1
            N=6; M=N+1

            ax.errorbar(J_list+shift, np.mean(SLED_mes[:, zi, idz, 1:], axis=0), yerr=np.std(SLED_mes[:, zi, idz, 1:], axis=0), 
                        fmt='o', color=c, ecolor=c, lw=1, markersize=2)
            
            patch = mpatches.Patch(color=c, label=f'z={z}')
            patchs.append(patch)
        #---------------------------
        ax.set_ylabel(r"$ \rm R_\mathrm{J_{up}-3} = B^{CO(J_{up}-J_{up}-1)}/B^{CO(3-2)} $")
        ax.legend(handles = patchs, loc= 'upper left',fontsize=6, frameon = False) #4.5
        ax.set_ylim(0,3)
        #---------------------------
        axr.set_ylim(-0.5,1)
        if(nslice==2.0): axr.set_ylim(-0.2,0.2)

        axr.plot((0,9), np.zeros(2), c='grey', lw=lw)
        axr.set_ylabel("relative \n difference")
        axr.set_xlabel("Quantum rotational number $\\rm J_{up}$")
        axr.set_xlim(0.5, 8.5)
        axr.tick_params(axis = "x", which='major', tickdir = "inout", top = True, color='k')
        fig.tight_layout(); fig.subplots_adjust(hspace=.0)
        for extension in ("png", "pdf"): plt.savefig(f"sled_dtype{dtype}_dz{dz}_nslice{nslice}_ppoints.{extension}", transparent=True)
        '''

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
        plt.ylabel('Contribution of SB objects \n to $\\rm B_{\\nu}^{CO(J_{up}-J_{up}-1}$ [%]')
        plt.tight_layout()
        for extension in ("png", "pdf"): plt.savefig(f"figs/sb_ms.{extension}", transparent=True)
        
    plt.show()

def co_sled_from_nsubfields(nslice, i_ref, z_list, dz_list, field_size, klim,
                            interlopers = None, allpoints=False, dtype='_with_interlopers', toembed=False):
    
    #---------------------------
    SLED_mes = np.zeros((12, len(z_list), len(dz_list), 9)) #12 subfields, redshift, redshift width, ref+8lines
    #for each redshift and subfield: 
    for idz, dz in enumerate(dz_list):
        for zi, z in enumerate(z_list):
            for nfield in range(12):

                if(toembed): embed()

                #-----------------
                #Load the dicts for transition of reference 
                file = f"dict_dir/dict_LIMgal_pySIDES_from_uchuu_ntile_{nfield}_z{z}_dz{dz}_{nslice}slices_{field_size}deg2_{line_list[i_ref]}{dtype}.p"
                dict= pickle.load( open(file, 'rb'))

                file = f"dict_dir/dict_LIMgal_pySIDES_from_uchuu_ntile_{nfield}_z{z}_dz{dz}_{nslice}slices_{field_size}deg2_{line_list[i_ref]}.p"
                d = pickle.load( open(file, 'rb'))

                #-----------------
                #Get the scales
                K = (dict["k"].to(u.rad**-1) * 2 * np.pi  /  d["Dc"]).value
                w = np.where( K <= klim)
                k_a = w[0][0]; k_e =  w[0][-1]
                if(k_e <= 1):  k_e = 2
                if(allpoints): born = [0,-1]
                else: born = [int(k_a),int(k_e+1)]

                intK, slopeK = print_scientific_notation( np.mean(K[k_e]) )
                #create patchs for legend
            
                #Get the power spectrum's mean and dispersion for the line of reference
                ref_k = (dict['pk_J-gal'][0][born[0]:born[1]].value - d["LIMgal_shot"][0].value)
                w = np.where(ref_k>0)
                ref = np.mean(ref_k[w])

                SLED_mes[nfield, zi, idz, 0] = ref
                #----------------
                #Get the power spectrum's mean and dispersion for the line of reference
                if(False):
                    plt.loglog(dict["k"].to(u.arcmin**-1), dict['pk_J-gal'][0].value - d["LIMgal_shot"][0].value, 'r')
                    plt.loglog(dict["k"].to(u.arcmin**-1), d['pk_J-gal'][0].value    - d["LIMgal_shot"][0].value, 'g')
                    plt.loglog(dict["k"].to(u.arcmin**-1)[born[0]:born[1]][w], ref_k[w], 'ob')
                    plt.show()

                for j, (J, rest_freq) in enumerate(zip(line_list[:8], rest_freq_list[:8])):

                    file = f"dict_dir/dict_LIMgal_pySIDES_from_uchuu_ntile_{nfield}_z{z}_dz{dz}_{nslice}slices_{field_size}deg2_{J}{dtype}.p"
                    dict = pickle.load( open(file, 'rb'))
                    file = f"dict_dir/dict_LIMgal_pySIDES_from_uchuu_ntile_{nfield}_z{z}_dz{dz}_{nslice}slices_{field_size}deg2_{J}.p"

                    d = pickle.load( open(file, 'rb'))
                    pk = dict['pk_J-gal'][0][born[0]:born[1]].value - d["LIMgal_shot"][0].value
                    w = np.where(pk>0)
                    SLED_mes[nfield, zi, idz, j+1] = (pk[w]).mean() / ref

                    if(False):
                        plt.loglog(dict["k"].to(u.arcmin**-1), dict['pk_J-gal'][0].value - d["LIMgal_shot"][0].value, 'r')
                        plt.loglog(d["k"].to(u.arcmin**-1), d['pk_J-gal'][0].value    - d["LIMgal_shot"][0].value, 'g')
                        plt.loglog(dict["k"].to(u.arcmin**-1)[born[0]:born[1]][w], (dict['pk_J-gal'][0][born[0]:born[1]].value - d["LIMgal_shot"][0].value)[w], 'ob')
                        plt.show()


                
    return SLED_mes
    
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
        for nslice, dz, toembed in zip(n_list, dz_list, (False, False, False)):
            plot_sled_fig(nslice, z_list, (dz,), args.recompute_sleds, toembed=toembed)
            plt.show()

    if(True): 
        for nslice, dz in zip(n_list, dz_list):
            contrib_ms_sb(z_list, (dz*(nslice*2+1),), args.recompute_sleds)
