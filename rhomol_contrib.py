import sys
import os
from gen_all_sizes_cubes_and_cat import *
from pysides.load_params import *
import time
import matplotlib
from IPython import embed
from progress.bar import Bar

def rhoh2(cat, Vslice,dz, alpha_co):
    nu_obs = 115.27120180 / (1+cat['redshift'])
    rho_Lprim =  np.sum(cat['ICO10'] * (cat["Dlum"]**2) * 3.25e7 / (1+cat["redshift"])**3 / nu_obs**2) / Vslice.value
    rhoh2 = rho_Lprim * alpha_co    
    return rhoh2 

params = load_params('PAR/cubes.par')

zmean = params['z_list']
dz = params['dz_list'][0]
zbins = [(z - dz/2, z + dz/2) for z in zmean]
Dc_bins = cosmo.comoving_distance(zbins)

dictfile = f"rhoh2_alphacoMS{params['alpha_co_ms']}_alphacoSB{params['alpha_co_sb']}.p"
if(not os.path.isfile(dictfile)):

    dict = {}
        
    for tile_sizeRA, tile_sizeDEC, N in params['tile_sizes']: 

        file = f"rhoh2_alphacoMS{params['alpha_co_ms']}_alphacoSB{params['alpha_co_sb']}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.p"
        if(not os.path.isfile(file)):
            
            dict_fields = {}
            dict_fields['z'] = zmean

            tile_size = tile_sizeRA*tile_sizeDEC
            field_size = tile_size * (np.pi/180.)**2
            rho_list = np.zeros((len(zmean), 3, N))

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
                    rho_list[i,1,l] = rhoh2(sb_cat, Vslice, dz, params['alpha_co_sb'])  #solar masses per Mpc cube
                    rho_list[i,2,l] = rho_list[i,1,l] + rho_list[i,2,l]
                    dict_fields[f'{l}']['MS'] = rho_list[i,0,l]
                    dict_fields[f'{l}']['SB'] = rho_list[i,1,l]
                    dict_fields[f'{l}']['TOT'] = rho_list[i,2,l]
                bar.next()

            for key, ikey in zip(('MS', 'SB', 'TOT'), (0,1,2)):
                dict_fields['0'][f'{key}_mean'] = np.mean(rho_list[:,ikey,:], axis=-1)
                dict_fields['0'][f'{key}_std']  = np.std(rho_list[:,ikey,:], axis=-1)

            pickle.dump(dict_fields, open(file, 'wb'))
            bar.finish
            print('')

            #plt.errorbar(zmean, np.mean(rho_list[:,0,:], axis=-1), yerr=np.std(rho_list[:,0,:], axis=-1), label=f'MS {tile_sizeRA}deg '+'$\\rm \\times$ '+f'{tile_sizeDEC}deg')
            #plt.errorbar(zmean, np.mean(rho_list[:,1,:], axis=-1), yerr=np.std(rho_list[:,1,:], axis=-1), label=f'SB {tile_sizeRA}deg '+'$\\rm \\times$ '+f'{tile_sizeDEC}deg')
            #plt.yscale('log')
            #plt.legend()
            #plt.show()

    else: dict_fields =  pickle.load( open(file, 'rb'))

    dict['{tile_sizeRA}deg_x_{tile_sizeDEC}deg'] = dict_fields
    pickle.dump(dict, open(dictfile, 'wb'))

else: dict = pickle.load( open(dictfile, 'rb'))
