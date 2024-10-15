import sys
import os
from gen_all_sizes_cubes_and_cat import *
from pysides.load_params import *
import time
import matplotlib
from IPython import embed
from progress.bar import Bar

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers

rest_freq_list = [115.27120180  *u.GHz* J_up for J_up in range(1, 9)]
rest_freq_list.append(freq_CI10); rest_freq_list.append(freq_CI21); 
rest_freq_list.append(freq_CII); 
line_list = ["CO{}{}".format(J_up, J_up - 1) for J_up in range(1, 9)]

def rhoh2(cat, Vslice,dz, alpha_co):

    nu_obs = 115.27120180 / (1+cat['redshift'])
    rho_Lprim =  np.sum(cat['ICO10'] * (cat["Dlum"]**2) * 3.25e7 / (1+cat["redshift"])**3 / nu_obs**2) / Vslice.value
    rhoh2 = rho_Lprim * alpha_co    

    return rhoh2 

def B_and_sn(cat, line, nu_rest, z, dz, field_size):
    
    nu_obs = nu_rest /(1+cat['redshift'])
    dnu    = dz*nu_obs/(1+z)
    vdelt = (cst.c * 1e-3) * dnu / nu_obs #km/s
    S = cat['I'+line] / vdelt  #Jy
    B = np.sum(S) / field_size

    return B 

params = load_params('PAR/cubes.par')

zmean = params['z_list']
dz = params['dz_list'][0]
zbins = [(z - dz/2, z + dz/2) for z in zmean]
Dc_bins = cosmo.comoving_distance(zbins)

dictfile = f"dict_dir/rhoh2_alphacoMS{params['alpha_co_ms']}_alphacoSB{params['alpha_co_sb']}.p"

if(not os.path.isfile(dictfile) ):

    dict = {}
        
    for tile_sizeRA, tile_sizeDEC, N in params['tile_sizes']: 

        file = f"dict_dir/rhoh2_alphacoMS{params['alpha_co_ms']}_alphacoSB{params['alpha_co_sb']}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.p"
        if(not os.path.isfile(file)):
            
            dict_fields = {}
            dict_fields['z'] = zmean

            tile_size = tile_sizeRA*tile_sizeDEC
            field_size = tile_size * (np.pi/180.)**2
            rho_list = np.zeros((len(zmean), 3, N))
            B_list = np.zeros((len(zmean), len(line_list), 3, N))
            Bratio_list_ttt = np.zeros((len(zmean), len(line_list), 3, N))
            Bratio_list = np.zeros((len(zmean), len(line_list), 3, N))

            bar = Bar(f'computing rhoH2(z) for {tile_sizeRA}deg x {tile_sizeDEC}deg', max=N)  

            for l in range(N):

                cat_subfield = Table.read( f'{params["output_path"]}/pySIDES_from_uchuu_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.fits' )
                cat_subfield = cat_subfield.to_pandas()
                dict_fields[f'{l}'] = {}

                for i, (z, (left_edge, right_edge), (left_Dc, right_Dc)) in enumerate(zip(zmean, zbins, Dc_bins)):
                    cat_bin = cat_subfield.loc[ (cat_subfield['redshift'] > left_edge) & (cat_subfield['redshift'] < right_edge)]
                    Vslice = field_size / 3 * (right_Dc**3-left_Dc**3)
                    ms_cat = cat_bin.loc[cat_bin['ISSB']==0]
                    sb_cat = cat_bin.loc[cat_bin['ISSB']==1]
                    rho_list[i,0,l] = rhoh2(ms_cat, Vslice, dz, params['alpha_co_ms'])  #solar masses per Mpc cube
                    if(len(sb_cat)>0): rho_list[i,1,l] = rhoh2(sb_cat, Vslice, dz, params['alpha_co_sb'])  #solar masses per Mpc cube
                    rho_list[i,2,l] = rho_list[i,1,l] + rho_list[i,0,l]

                    for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):

                        B_list[i,j,1,l] = B_and_sn(sb_cat, line, rest_freq, z, dz, field_size)
                        B_list[i,j,0,l] = B_and_sn(ms_cat, line, rest_freq, z, dz, field_size)
                        B_list[i,j,2,l] = B_list[i,j,0,l] + B_list[i,j,1,l]

                    for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):

                        Bratio_list_ttt[i,j,0,l] = B_list[i,j,0,l] / B_list[i,2,2,l]
                        Bratio_list_ttt[i,j,1,l] = B_list[i,j,1,l] / B_list[i,2,2,l]
                        Bratio_list_ttt[i,j,2,l] = B_list[i,j,2,l] / B_list[i,2,2,l]

                        Bratio_list[i,j,0,l] = B_list[i,j,0,l] / B_list[i,0,2,l]
                        Bratio_list[i,j,1,l] = B_list[i,j,1,l] / B_list[i,0,2,l]
                        Bratio_list[i,j,2,l] = B_list[i,j,2,l] / B_list[i,0,2,l]

                for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):

                    dict_fields[f'{l}'][line] = {}
                    dict_fields[f'{l}'][line]['B_MS'] = B_list[:,j,0,l]
                    dict_fields[f'{l}'][line]['B_SB'] = B_list[:,j,1,l]
                    dict_fields[f'{l}'][line]['B_TOT'] = B_list[:,j,2,l]
                    dict_fields[f'{l}'][line]['ratio_B_SB_MS'] = B_list[:,j,1,l] / B_list[:,j,0,l]

                    dict_fields[f'{l}'][line]['B_MS/B_CO32'] = Bratio_list_ttt[:,j,0,l]
                    dict_fields[f'{l}'][line]['B_SB/B_CO32'] = Bratio_list_ttt[:,j,1,l]
                    dict_fields[f'{l}'][line]['B_TOT/B_CO32'] = Bratio_list_ttt[:,j,2,l]

                    dict_fields[f'{l}'][line]['B_MS/B_CO10'] = Bratio_list[:,j,0,l]
                    dict_fields[f'{l}'][line]['B_SB/B_CO10'] = Bratio_list[:,j,1,l]
                    dict_fields[f'{l}'][line]['B_TOT/B_CO10'] = Bratio_list[:,j,2,l]

                dict_fields[f'{l}']['MS'] = rho_list[:,0,l]
                dict_fields[f'{l}']['SB'] = rho_list[:,1,l]
                dict_fields[f'{l}']['TOT'] = rho_list[:,2,l]

                bar.next()

        
            dict_fields[f'ratio_rho_sb_ms_mean'] = np.mean(rho_list[:,1,:]/rho_list[:,0,:], axis=-1)
            dict_fields[f'ratio_rho_sb_ms_std']  = np.std(rho_list[:,1,:]/rho_list[:,0,:], axis=-1)

            for key, ikey in zip(('MS', 'SB', 'TOT'), (0,1,2)):

                if(key != 'TOT'):

                    dict_fields[f'ratio_rho_{key}_TOT_mean'] = np.mean(rho_list[:,ikey,:]/rho_list[:,2,:], axis=-1)
                    dict_fields[f'ratio_rho_{key}_TOT_std']  = np.std(rho_list[:,ikey,:]/rho_list[:,2,:], axis=-1)

                dict_fields[f'{key}_mean'] = np.mean(rho_list[:,ikey,:], axis=-1)
                dict_fields[f'{key}_std']  = np.std(rho_list[:,ikey,:], axis=-1)


                for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):

                    if(key != 'TOT'):

                        dict_fields[f'ratio_B_{key}_{line}_TOT_mean'] = np.mean(B_list[:,j,ikey,:]/B_list[:,j,2,:], axis=-1)
                        dict_fields[f'ratio_B_{key}_{line}_TOT_std']  = np.std(B_list[:,j,ikey,:]/B_list[:,j,2,:], axis=-1)

                    dict_fields[f'B_{key}_{line}_mean'] = np.mean(B_list[:,j,ikey,:], axis=-1)
                    dict_fields[f'B_{key}_{line}_std'] = np.std(B_list[:,j,ikey,:], axis=-1)
                
                    dict_fields[f'B_{key}_{line}/B_CO32_mean'] = np.mean(Bratio_list_ttt[:,j,ikey,:], axis=-1)
                    dict_fields[f'B_{key}_{line}/B_CO32_std'] = np.std(Bratio_list_ttt[:,j,ikey,:], axis=-1)

                    dict_fields[f'B_{key}_{line}/B_CO10_mean'] = np.mean(Bratio_list[:,j,ikey,:], axis=-1)
                    dict_fields[f'B_{key}_{line}/B_CO10_std'] = np.std(Bratio_list[:,j,ikey,:], axis=-1)

                    dict_fields[f'ratio_B_SB_MS_{line}_mean'] = np.mean(B_list[:,j,1,:] / B_list[:,j,0,:], axis=-1)
                    dict_fields[f'ratio_B_SB_MS_{line}_std'] = np.std(B_list[:,j,1,:] / B_list[:,j,0,:], axis=-1)

            pickle.dump(dict_fields, open(file, 'wb'))
            bar.finish
            print('')

        else: dict_fields =  pickle.load( open(file, 'rb'))

        dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'] = dict_fields
        pickle.dump(dict, open(dictfile, 'wb'))

else: dict = pickle.load( open(dictfile, 'rb'))

# --- SLED --- 

colors_co = ('orange', 'r', 'b', 'cyan', 'g', 'purple', 'magenta', 'grey',)

#Bnu vs z of lines
if(True): 

    for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):

        BS = 7; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
        fig, (ax, axs, axr, axrr, axrrr) = plt.subplots(5, 1, sharex=True, sharey = 'row', 
                                        gridspec_kw={'height_ratios': [2,1,1,1,1]}, 
                                        figsize=(5,4.5), dpi = 200)
    
        for tile_sizeRA, tile_sizeDEC, _ in params['tile_sizes']: 
            for key, c, ls in zip(("MS", 'SB'), ('r','g'), ('solid', '--')):
                x = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg']['z']
                y = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_{key}_{line}_mean']
                dy = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_{key}_{line}_std']
                if(tile_sizeRA == 3): ax.errorbar(x,y, c='k',ls=ls)
                ax.fill_between(x,y-dy,y+dy, color=colors_co[j], alpha=0.2)
        
        for tile_sizeRA, tile_sizeDEC, _ in params['tile_sizes']: 
            x = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg']['z']
            y = 100*dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'ratio_B_SB_MS_{line}_mean']
            dy =100* dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'ratio_B_SB_MS_{line}_std']
            if(tile_sizeRA == 3): 
                axs.errorbar(x,y, c='k',ls=ls)
                if(j==0): print(y)
            axs.fill_between(x,y-dy,y+dy, color=colors_co[j], alpha=0.2)
        axs.set_ylabel('$\\rm B^{SB}_{\\nu} / B^{MS}_{\\nu}$ [%]')


        for tile_sizeRA, tile_sizeDEC, _ in params['tile_sizes']: 
            x = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg']['z']
            y = 100*dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'ratio_B_SB_{line}_TOT_mean']
            dy =100* dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'ratio_B_SB_{line}_TOT_std']
            if(tile_sizeRA == 3): axr.errorbar(x,y, c='k',ls=ls)
            axr.fill_between(x,y-dy,y+dy, color=colors_co[j], alpha=0.2)

        for tile_sizeRA, tile_sizeDEC, _ in params['tile_sizes']: 
            x = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg']['z']
            yA = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_SB_{line}/B_CO32_mean']
            dyA =dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_SB_{line}/B_CO32_std']
            if(tile_sizeRA == 3): axrr.errorbar(x,yA, c='g',ls=':')
            axrr.fill_between(x,yA-dyA,yA+dyA, color='g', alpha=0.2)

            yB = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_MS_{line}/B_CO32_mean']
            dyB =dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_MS_{line}/B_CO32_std']
            if(tile_sizeRA == 3): axrr.errorbar(x,yB, c='r',ls='--')
            axrr.fill_between(x,yB-dyB,yB+dyB, color='r', alpha=0.2)

            y = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_TOT_{line}/B_CO32_mean']
            dy =dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_TOT_{line}/B_CO32_std']
            if(tile_sizeRA == 3): axrr.errorbar(x,y, c='b',ls='--')
            axrr.fill_between(x,y-dy,y+dy, color='b', alpha=0.2)

            patchs = []
            patch = mlines.Line2D([], [], color='r', linestyle='solid', label='MS'); patchs.append(patch)
            patch = mlines.Line2D([], [], color='g', linestyle='solid', label='SB'); patchs.append(patch)
            patch = mlines.Line2D([], [], color='b', linestyle='solid', label='MS+SB'); patchs.append(patch)

            axrr.legend(handles = patchs, frameon=False)

            #if(tile_sizeRA == 3): axrr.errorbar(x,yB+yA, c='k')

        axrr.set_ylabel('$\\rm B_{\\nu} / B^{CO32,(SB+MS)}_{\\nu}$')


        for tile_sizeRA, tile_sizeDEC, _ in params['tile_sizes']: 
            x = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg']['z']
            yA = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_SB_{line}/B_CO10_mean']
            dyA =dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_SB_{line}/B_CO10_std']
            if(tile_sizeRA == 3): axrrr.errorbar(x,yA, c='g',ls=':')
            axrrr.fill_between(x,yA-dyA,yA+dyA, color='g', alpha=0.2)

            yB = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_MS_{line}/B_CO10_mean']
            dyB =dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_MS_{line}/B_CO10_std']
            if(tile_sizeRA == 3): axrrr.errorbar(x,yB, c='r',ls='--')
            axrrr.fill_between(x,yB-dyB,yB+dyB, color='r', alpha=0.2)

            y = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_TOT_{line}/B_CO10_mean']
            dy =dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'B_TOT_{line}/B_CO10_std']
            if(tile_sizeRA == 3): axrrr.errorbar(x,y, c='b',ls='--')
            axrrr.fill_between(x,y-dy,y+dy, color='b', alpha=0.2)

        axrrr.set_ylabel('$\\rm B_{\\nu} / B^{CO10,(SB+MS)}_{\\nu}$')

        axrrr.set_ylim(0,7.5)
        axrr.set_ylim(0,7.5)

        patchs = []
        patch = mlines.Line2D([], [], color='k', linestyle='solid',  label='B$\\rm \\nu$ MS'); patchs.append(patch)
        patch = mlines.Line2D([], [], color='k', linestyle='--',     label='B$\\rm \\nu$ SB'); patchs.append(patch)
        ax.set_yscale('log')
        axrr.set_xlabel('redshift')
        ax.set_ylabel('B$\\rm \\nu$ [Jy/sr]'+f' of {line}')
        axr.set_ylabel('$\\rm B^{SB}_{\\nu} / B^{MS+SB}_{\\nu}$ [%] ')
        ax.legend(handles = patchs, frameon=False)
        fig.tight_layout(); fig.subplots_adjust(hspace=.0)
        plt.savefig(f'{line}_SB_and_MS_Bnu.png', transparent=True)

    plt.show()


#Rho plot
if(False):

    BS = 10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig, (ax, axr,axrr) = plt.subplots(3, 1, sharex=True, sharey = 'row', 
                                  gridspec_kw={'height_ratios': [2,1,1]}, 
                                  figsize=(5,4.5), dpi = 200)

    for tile_sizeRA, tile_sizeDEC, _ in params['tile_sizes']: 
        for key, c, ls in zip(("MS", 'SB'), ('r','g'), ('solid', '--')):
            
            x = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg']['z']
            y = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'{key}_mean']
            dy = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'{key}_std']
            if(tile_sizeRA == 3): ax.errorbar(x,y, c='k',ls=ls)
            ax.fill_between(x,y-dy,y+dy, color=c, alpha=0.2)
        
        y = 100*(dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg']['ratio_rho_SB_TOT_mean'])
        dy = 100*(dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg']['ratio_rho_SB_TOT_std'])
        if(tile_sizeRA == 3): axr.errorbar(x,y, ls='--',c='k')
        axr.fill_between(x,y-dy,y+dy, color='g', alpha=0.2)

        y = 100*(dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg']['ratio_rho_sb_ms_mean'])
        dy = 100*(dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg']['ratio_rho_sb_ms_std'])
        if(tile_sizeRA == 3): axrr.errorbar(x,y, ls='--',c='k'); print(y)
        axrr.fill_between(x,y-dy,y+dy, color='g', alpha=0.2)

    patchs = []
    patch = mlines.Line2D([], [], color='k', linestyle='solid',  label='$\\rm \\rho_{H2}$ MS'); patchs.append(patch)
    patch = mlines.Line2D([], [], color='k', linestyle='--',     label='$\\rm \\rho_{H2}$ SB'); patchs.append(patch)
    patch = mpatches.Patch(color='r', label='field-to-field variance for MS' ); patchs.append(patch)
    patch = mpatches.Patch(color='g', label='field-to-field variance for SB' ); patchs.append(patch)
    
    ax.set_title('$\\rm \\alpha_{CO}^{MS}=$'+f'{params["alpha_co_ms"]}, '+'$\\rm \\alpha_{CO}^{SB}=$'+f'{params["alpha_co_sb"]} '+
            '[$\\rm M_{\\odot}.(K.km.s^{-1}.pc^2)^{-1}$]' )
    ax.set_yscale('log')
    axr.set_xlabel('redshift')
    ax.set_ylabel('$\\rm \\rho_{H2} [M_{\\odot}.Mpc^{-3}]$')
    axr.set_ylabel('$\\rm \\rho^{SB}_{H2} / \\rho^{(SB+MS)}_{H2}$ [%]')
    ax.legend(handles = patchs, frameon=False)
    fig.tight_layout(); fig.subplots_adjust(hspace=.0)
    plt.savefig(f'rhoh2_alphaCOMS_{params["alpha_co_ms"]}_alphaCOSB_{params["alpha_co_sb"]}.pdf',transparent=True)
    plt.show()
