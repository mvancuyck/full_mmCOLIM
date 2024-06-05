from fcts import * 
from gen_all_sizes_cubes import * 
from functools import partial
from multiprocessing import Pool, cpu_count
from matplotlib import gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
import argparse
from pysides.load_params import *

line_list_fancy = ["CO({}-{})".format(J_up, J_up - 1) for J_up in range(1, 9)]
#line_list.append('CI10'); line_list.append('CI21'); 
line_list_fancy.append('[CII]')

def CO10_LF(z_list, dz, alpha_co = 3.6):

    simu, cat, dirpath, fs = load_cat()

    patchs=[]
    for tile_size, c in zip((1.5,9),('k', 'r', 'g')): #(1.5,9):
        patch = mlines.Line2D([], [], color=c, linestyle="None", marker="o", label=f'{tile_size}'+'$\\rm deg^2$'); patchs.append(patch)
                                                            
        ragrid=np.arange(cat['ra'].min(),cat['ra'].max(),np.sqrt(tile_size))
        decgrid=np.arange(cat['dec'].min(),cat['dec'].max(),np.sqrt(tile_size))
        grid=np.array(np.meshgrid(ragrid,decgrid))

        ra_index = np.arange(0,len(ragrid)-1,1)
        dec_index = np.arange(0,len(decgrid)-1,1)
        ra_grid, dec_grid = np.meshgrid(ra_index, dec_index)
        # Flatten the grids and stack them into a single array
        coords = np.stack((ra_grid.flatten(), dec_grid.flatten()), axis=1)

        rho_field = []

        for z in z_list:

            catz = cat.loc[ (cat['redshift'] >= z-dz/2) & (cat['redshift'] < z+dz/2)]
            rho = []

            for l, (ira, idec) in enumerate(coords):
                if(l+1>200): continue

                cat_subfield = catz.loc[(catz['ra']>=grid[0,idec,ira])&(catz['ra']<grid[0,idec,ira+1])&(catz['dec']>=grid[1,idec,ira])&(catz['dec']<grid[1,idec+1,ira])]

                nu_obs = 115.27120180 / (1+cat_subfield['redshift'])
                Lprim =  np.sum(cat_subfield['ICO10'] * (cat_subfield["Dlum"]**2) * 3.25e7 / (1+cat_subfield["redshift"])**3 / nu_obs**2)
                Vslice = (tile_size*u.deg**2).to(u.sr) / 3 * (cosmo.comoving_distance(z+dz/2)**3-cosmo.comoving_distance(z-dz/2)**3)
                rho.append( alpha_co * Lprim / Vslice.value )


            rho_field.append(np.asarray(rho))
            
            plt.errorbar(z, np.asarray(rho).mean(), yerr=np.asarray(rho).std(), fmt='o', color=c, ecolor=c)

        dict = {f'rho_{tile_size}deg2':np.asarray(rho_field)}
        pickle.dump(dict, open(f'rho_{tile_size}deg2.p', 'wb'))

    plt.legend(handles=patchs, bbox_to_anchor=(1.4,1), frameon=False)
    plt.ylabel('$\\rm \\rho_{H2} [M_{\\odot} Mpc^{-3}]$')
    plt.xlabel('redshift')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('rho_sides.png', dpi=200, transparent=True)
    plt.show()

tim_params = load_params('PAR/cubes.par')
z_list = tim_params['z_list']
dz_list = tim_params['dz_list']
n_list = tim_params['n_list']

for i, (dz, nslice) in enumerate(zip(dz_list, n_list)): 
    CO10_LF(z_list, 0.1)
