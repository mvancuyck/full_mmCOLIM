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

def mol_gas_density(cat, Vslice, alpha_co):
    nu_obs = 115.27120180 / (1+cat['redshift'])
    Lprim =  np.sum(cat['ICO10'] * (cat["Dlum"]**2) * 3.25e7 / (1+cat["redshift"])**3 / nu_obs**2)
    return alpha_co * Lprim / Vslice.value       
    
params_sides = load_params('PAR/SIDES_from_original_with_fir_lines.par')
params = load_params('PAR/cubes.par')
params['output_path'] = '/net/CONCERTO/home/mvancuyck/TIM_pysides_user_friendly/OUTPUT_TIM_CUBES_FROM_UCHUU/'

z_list = params['z_list']
dz_list = params['dz_list']
n_list = params['n_list']

dict = {}

for tile_sizeRA, tile_sizeDEC in params['tile_sizes']: 
    if(tile_sizeDEC == 9): continue

    dict_fieldsize = {}
    
    # List files matching the pattern
    files = sorted_files_by_n(params["output_path"], ((tile_sizeRA, tile_sizeDEC),))
    field_size = (tile_sizeRA * tile_sizeDEC *u.deg**2).to(u.sr)
    
    for l, file in enumerate(files):
        
        cat = Table.read(params["output_path"]+file)
        cat = cat.to_pandas()

        dict_tile = {}

        for z in z_list: 

            Dz = dz_list[0] * n_list[0]
            Vslice = field_size / 3 * (cosmo.comoving_distance(z+Dz/2)**3-cosmo.comoving_distance(z-Dz/2)**3)

            rho_MS = mol_gas_density(cat.loc[cat['issb'] == False], Vslice, params['alpha_co_ms'])
            rho_SB = mol_gas_density(cat.loc[cat['issb'] == True],  Vslice, params['alpha_co_sb'])

            dict_tile[f'rho_mol_MS_at_{z}'] = rho_MS
            dict_tile[f'rho_mol_SB_at_{z}'] = rho_SB
    
        dict_fieldsize[f'{l}'] = dict_tile

    dict[f'{tile_sizeRA}deg_x_{tile_sizeDEC}deg'] = dict_fieldsize

pickle.dump(dict, open(f"dict_dir/rhomol_alphacoMS{params['alpha_co_ms']}_alphaCOSB{params['alpha_co_sb']}.p", 'wb'))


 