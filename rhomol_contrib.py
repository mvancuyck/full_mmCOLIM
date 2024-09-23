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
    #rho_Lprim =  np.sum(cat['ICO10'] * (cat["Dlum"]**2) * 3.25e7 / (1+cat["redshift"])**3 / nu_obs**2)
    dnu=dz*nu_obs/(1+z)
    vdelt = (cst.c * 1e-3) * dnu / nu_obs #km/s
    S = cat['ICO10'] / vdelt  #Jy
    rhoL =  S * ((4*np.pi*115.27120180e9*cosmo.H(cat['redshift']))/(4e7 *cst.c*1e-3)) / Vslice.value #Lsolar/Mpc3
    Lprim = rhoL * 3.11e10/(nu_obs*(1+cat['redshift']))**3
    rhoh2 = Lprim * alpha_co
    #-------------------------
    
    return np.sum(Lprim) 

params = load_params('PAR/cubes.par')

zmean = params['z_list']
dz = params['dz_list'][0]
zbins = [(z - dz/2, z + dz/2) for z in zmean]
Dc_bins = cosmo.comoving_distance(zbins)

for tile_sizeRA, tile_sizeDEC, N in params['tile_sizes']: 

    tile_size = tile_sizeRA*tile_sizeDEC
    field_size = tile_size * (np.pi/180.)**2
    SFRD_list = np.zeros((len(zmean), N))

    for l in range(N):

        if l >= 120: break  # Exit both loops
        #cat_subfield=cat.loc[(cat['ra']>=grid[0,idec,ira])&(cat['ra']<grid[0,idec,ira+1])&(cat['dec']>=grid[1,idec,ira])&(cat['dec']<grid[1,idec+1,ira])]
        cat_subfield = Table.read( f'{params["output_path"]}/pySIDES_from_bolshoi_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg.fits' )
        cat_subfield = cat_subfield.to_pandas()

        for i, (z, (left_edge, right_edge), (left_Dc, right_Dc)) in enumerate(zip(zmean, zbins, Dc_bins)):
            cat_bin = cat_subfield.loc[ (cat_subfield['redshift'] > left_edge) & (cat_subfield['redshift'] < right_edge)]
            Vslice = field_size / 3 * (right_Dc**3-left_Dc**3)
            SFRD_list[i,l] = rhoh2(cat_bin, Vslice, dz, params['alpha_co_ms'])  #solar masses per year per Mpc cube

    plt.errorbar(zmean, np.mean(SFRD_list, axis=-1), yerr=np.std(SFRD_list, axis=-1))

plt.show()