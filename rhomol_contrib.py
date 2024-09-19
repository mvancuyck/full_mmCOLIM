import sys
import os
from gen_all_sizes_cubes_and_cat import *
from pysides.load_params import *
import time
import matplotlib
from IPython import embed
from progress.bar import Bar

def mol_gas_density(cat, dz, field_size, alpha_co):

    nu_obs = 115.27120180 / (1+cat['redshift'])
    dnu=dz*nu_obs/(1+z)
    vdelt = (cst.c * 1e-3) * dnu / nu_obs #km/s
    S = cat['ICO10'] / vdelt  #Jy
    rhoL = ((4*np.pi*115.27120180e9*cosmo.H(cat['redshift']))/(4e7 *cst.c*1e-3)) #Lsolar/Mpc3
    Lprim = 3.11e10/(nu_obs*(1+cat['redshift']))**3
    rhoh2 = np.sum(S*rhoL*Lprim*alpha_co) / field_size.value
    #-------------------------
    return rhoh2
    
params = load_params('PAR/cubes.par')
params['output_path'] = '/net/CONCERTO/home/mvancuyck/TIM_pysides_user_friendly/OUTPUT_TIM_CUBES_FROM_UCHUU/'

z_list = params['z_list']
dz_list = params['dz_list']
n_list = params['n_list']

file1 = f"dict_dir/rhomol_alphacoMS{params['alpha_co_ms']}_alphaCOSB{params['alpha_co_sb']}.p"


if( not os.path.isfile(file1) ):
    dict = {}

    for tile_sizeRA, tile_sizeDEC, Nsimu in params['tile_sizes']: 

        # List files matching the pattern

        file = f"dict_dir/rhomol_alphacoMS{params['alpha_co_ms']}_alphaCOSB{params['alpha_co_sb']}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.p"
        if( not os.path.isfile(file) ):
            print('')

            dict_fieldsize = {}
            
            bar = Bar(f'computing rho_mol(z) for {tile_sizeRA}x{tile_sizeDEC}deg2', max=Nsimu)  

            tab = np.zeros((Nsimu, len(z_list), 3 ))

            for l in range(Nsimu):
                
                cat = Table.read(params["output_path"]+f"pySIDES_from_uchuu_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.fits")
                cat = cat.to_pandas()
                dict_tile = {}

                for iz, z in enumerate(z_list): 

                    Dz = dz_list[0] * n_list[0]
                    Vslice = (tile_sizeRA * tile_sizeDEC *u.deg**2).to(u.sr) / 3 * (cosmo.comoving_distance(z+Dz/2)**3-cosmo.comoving_distance(z-Dz/2)**3)
                    catbin = cat.loc[ (cat['redshift']>= z-Dz/2) & (cat['redshift']<= z+Dz/2)]
                    dict_tile[f'rho_mol_MS_at_z{z}'] = mol_gas_density(catbin, Dz, Vslice, params['alpha_co_ms']) #.loc[catbin['ISSB'] == 0]
                    tab[l,iz,0] = dict_tile[f'rho_mol_MS_at_z{z}']

                    #rho_SB = mol_gas_density(catbin.loc[catbin['ISSB'] == 1], Dz, field_size, params['alpha_co_sb'])
                    #dict_tile[f'rho_mol_SB_at_z{z}'] = rho_SB
                    #dict_tile[f'rho_mol_TOT_at_z{z}'] = rho_SB+rho_MS
                    #tab[l,iz,1] = rho_SB
                    #tab[l,iz,2] = rho_SB+rho_MS

                dict_fieldsize[f'tile_{l}'] = dict_tile
                bar.next() 
            
            dict_fieldsize['tile_0'][f'MS_mean'] = np.mean(tab[:,:,0], axis = (0))
            dict_fieldsize['tile_0'][f'MS_median'] = np.median(tab[:,:,0], axis = (0))
            dict_fieldsize['tile_0'][f'MS_std'] = np.std(tab[:,:,0], axis = (0))
            dict_fieldsize['redshift'] = z_list
            #dict_fieldsize['tile_0'][f'SB_mean'] = np.mean(tab[:,:,1], axis = (0))
            #dict_fieldsize['tile_0'][f'SB_median'] = np.median(tab[:,:,1], axis = (0))
            #dict_fieldsize['tile_0'][f'SB_std'] = np.std(tab[:,:,1], axis = (0))
            #dict_fieldsize['tile_0'][f'TOT_mean'] = np.mean(tab[:,:,2], axis = (0))
            #dict_fieldsize['tile_0'][f'TOT_median'] = np.median(tab[:,:,2], axis = (0))
            #dict_fieldsize['tile_0'][f'TOT_std'] = np.std(tab[:,:,2], axis = (0))
            bar.finish
            pickle.dump(dict_fieldsize, open(file, 'wb'))
        else: 
            dict_fieldsize = pickle.load( open(file, 'rb'))

        dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'] = dict_fieldsize
        dict['redshift'] = z_list

    pickle.dump(dict, open(file1, 'wb'))

else: dict = pickle.load( open(file1, 'rb'))

if(True):
    
    z = dict['redshift']
    for key, c  in zip(('9deg_x_9deg', '1.5deg_x_1.5deg'),('g','b')): 
        m =   dict[key]['tile_0']['MS_mean']
        std = dict[key]['tile_0']['MS_std']
        plt.plot(z, m, 'g'); plt.fill_between(z,m-std, m+std, color='g', alpha=0.2)
    plt.yscale('log')

    plt.figure()
    plt.plot(z, dict['9deg_x_9deg']['tile_0']['MS_mean'] / dict['1.5deg_x_1.5deg']['tile_0']['MS_mean'], 'r')
    plt.show()