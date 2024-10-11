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

            bar = Bar(f'computing rhoH2(z) for {tile_sizeRA}deg x {tile_sizeDEC}deg', max=N)  

            for l in range(N):

                cat_subfield = Table.read( f'{params["sides_cat_path"]}/pySIDES_from_uchuu_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.fits' )
                cat_subfield = cat_subfield.to_pandas()
                dict_fields[f'{l}'] = {}

                for i, (z, (left_edge, right_edge), (left_Dc, right_Dc)) in enumerate(zip(zmean, zbins, Dc_bins)):
                    cat_bin = cat_subfield.loc[ (cat_subfield['redshift'] > left_edge) & (cat_subfield['redshift'] < right_edge)]
                    Vslice = field_size / 3 * (right_Dc**3-left_Dc**3)
                    ms_cat = cat_bin.loc[cat_bin['ISSB']==0]
                    sb_cat = cat_bin.loc[cat_bin['ISSB']==1]
                    rho_list[i,0,l] = rhoh2(ms_cat, Vslice, dz, params['alpha_co_ms'])  #solar masses per Mpc cube
                    rho_list[i,1,l] = rhoh2(sb_cat, Vslice, dz, params['alpha_co_sb'])  #solar masses per Mpc cube
                    rho_list[i,2,l] = rho_list[i,1,l] + rho_list[i,0,l]

                    for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):

                        B_list[i,j,1,l] = B_and_sn(sb_cat, line, rest_freq, z, dz, field_size)
                        B_list[i,j,0,l] = B_and_sn(ms_cat, line, rest_freq, z, dz, field_size)
                        B_list[i,j,2,l] = B_list[i,j,0,l] + B_list[i,j,1,l]

                for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):

                    dict_fields[f'{l}'][line] = {}
                    dict_fields[f'{l}'][line]['B_MS'] = B_list[:,j,0,l]
                    dict_fields[f'{l}'][line]['B_SB'] = B_list[:,j,1,l]
                    dict_fields[f'{l}'][line]['B_TOT'] = B_list[:,j,2,l]

                dict_fields[f'{l}']['MS'] = rho_list[:,0,l]
                dict_fields[f'{l}']['SB'] = rho_list[:,1,l]
                dict_fields[f'{l}']['TOT'] = rho_list[:,2,l]

                bar.next()

            for key, ikey in zip(('MS', 'SB', 'TOT'), (0,1,2)):

                dict_fields[f'{key}_mean'] = np.mean(rho_list[:,ikey,:], axis=-1)
                dict_fields[f'{key}_std']  = np.std(rho_list[:,ikey,:], axis=-1)

                for j, (line, rest_freq) in enumerate(zip(line_list, rest_freq_list)):

                    dict_fields[f'{l}'][line]['B_{key}_mean'] = np.mean(B_list[:,j,ikey,:], axis=-1)
                    dict_fields[f'{l}'][line]['B_{key}_std'] = np.std(B_list[:,j,ikey,:], axis=-1)

            pickle.dump(dict_fields, open(file, 'wb'))
            bar.finish
            print('')

        else: dict_fields =  pickle.load( open(file, 'rb'))

        dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'] = dict_fields
        pickle.dump(dict, open(dictfile, 'wb'))

else: dict = pickle.load( open(dictfile, 'rb'))


for tile_sizeRA, tile_sizeDEC, _ in params['tile_sizes']: 
    for key, c, ls in zip(("MS", 'SB'), ('r','g'), ('solid', '--')):
        
        x = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg']['z']
        y = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'{key}_mean']
        dy = dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'][f'{key}_std']
        if(tile_sizeRA == 3): plt.errorbar(x,y, c='k',ls=ls)
        plt.fill_between(x,y-dy,y+dy, color=c, alpha=0.2)

patchs = []
patch = mlines.Line2D([], [], color='k', linestyle='solid',  label='$\\rm \\rho_{H2}$ MS'); patchs.append(patch)
patch = mlines.Line2D([], [], color='k', linestyle='--',     label='$\\rm \\rho_{H2}$ SB'); patchs.append(patch)
patch = mpatches.Patch(color='r', label='field-to-field variance for MS' ); patchs.append(patch)
patch = mpatches.Patch(color='g', label='field-to-field variance for SB' ); patchs.append(patch)
plt.title('$\\rm \\alpha_{CO}^{MS}=$'+f'{params["alpha_co_ms"]}, '+'$\\rm \\alpha_{CO}^{SB}=$'+f'{params["alpha_co_sb"]} '+
          '[$\\rm M_{\\odot}.(K.km.s^{-1}.pc^2)^{-1}$]' )
plt.yscale('log')
plt.xlabel('redshift')
plt.ylabel('$\\rm \\rho_{H2} [M_{\\odot}.Mpc^{-3}]$')
plt.legend(handles = patchs)
plt.show()
