import sys
import os
from fcts import * 
from gen_all_sizes_cubes_and_cat import *
from pysides.make_cube import *
from pysides.load_params import *
from pysides.gen_outputs import *
import argparse
import time
import matplotlib
from IPython import embed
from progress.bar import Bar

def mol_gas_density(cat, Vslice, alpha_co):
    nu_obs = 115.27120180 / (1+cat['redshift'])
    Lprim =  np.sum(cat['ICO10'] * (cat["Dlum"]**2) * 3.25e7 / (1+cat["redshift"])**3 / nu_obs**2)
    return alpha_co * Lprim / Vslice.value       
    
params = load_params('PAR/cubes.par')
params['output_path'] = '/net/CONCERTO/home/mvancuyck/TIM_pysides_user_friendly/OUTPUT_TIM_CUBES_FROM_UCHUU/'

z_list = params['z_list']
dz_list = params['dz_list']
n_list = params['n_list']

dict = {}

for tile_sizeRA, tile_sizeDEC, Nsimu in params['tile_sizes']: 

    dict_fieldsize = {}
    
    # List files matching the pattern
    field_size = (tile_sizeRA * tile_sizeDEC *u.deg**2).to(u.sr)
    
    bar = Bar(f'computing rho_mol(z) for {tile_sizeRA}x{tile_sizeDEC}deg2', max=Nsimu)  

    for l in range(Nsimu):
        
        cat = Table.read(params["output_path"]+f"pySIDES_from_uchuu_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.fits")
        if(not 'ISSB' in list(cat.columns)): continue
        cat = cat.to_pandas()
        dict_tile = {}

        for z in z_list: 

            Dz = dz_list[0] * n_list[0]
            Vslice = field_size / 3 * (cosmo.comoving_distance(z+Dz/2)**3-cosmo.comoving_distance(z-Dz/2)**3)
        
            rho_MS = mol_gas_density(cat.loc[cat['ISSB'] == 0], Vslice, params['alpha_co_ms'])
            rho_SB = mol_gas_density(cat.loc[cat['ISSB'] == 1],  Vslice, params['alpha_co_sb'])
            dict_tile[f'rho_mol_MS_at_{z}'] = rho_MS
            dict_tile[f'rho_mol_SB_at_{z}'] = rho_SB
    
        dict_fieldsize[f'{l}'] = dict_tile
        bar.next() 

    bar.finish
    pickle.dump(dict, open(f"dict_dir/rhomol_alphacoMS{params['alpha_co_ms']}_alphaCOSB{params['alpha_co_sb']}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.p", 'wb'))
    dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'] = dict_fieldsize

pickle.dump(dict, open(f"dict_dir/rhomol_alphacoMS{params['alpha_co_ms']}_alphaCOSB{params['alpha_co_sb']}.p", 'wb'))
#dict = pickle.load( open(file, 'rb'))


 