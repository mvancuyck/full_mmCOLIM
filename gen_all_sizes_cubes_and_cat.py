import sys
import os
from fcts import * 
from pysides.make_cube import *
from pysides.load_params import *
from pysides.gen_outputs import *


import argparse
import time
import matplotlib
_args = None 

freq_CII = 1900.53690000 * u.GHz
freq_CI10 = 492.16 *u.GHz
freq_CI21 = 809.34 * u.GHz
rest_freq_list = [115.27120180  *u.GHz* J_up for J_up in range(1, 9)]
rest_freq_list.append(freq_CI10); rest_freq_list.append(freq_CI21); 
rest_freq_list.append(freq_CII); 
line_list = ["CO{}{}".format(J_up, J_up - 1) for J_up in range(1, 9)]
line_list.append('CI10'); line_list.append('CI21'); 
line_list.append('CII_de_Looze')

#Paralelisation on 24 nodes
#----


def load_cat():

    import matplotlib
    matplotlib.use("Agg")
    start = time.time()
    
    cats_dir_path='/net/CONCERTO/SIDES/PYSIDES_UCHUU_OUTPUTS/vpeak_complete/'
    
    filenames=[]
    with os.scandir(cats_dir_path) as it:
        for entry in it:
            filenames.append(cats_dir_path+entry.name)
        filenames=np.array(np.unique(filenames,axis=0))
        
    print('Loading the total data frame ...')
    df = vx.open_many(filenames)

    print('Converting the dataframe to pandas df ...')
    cat = df.to_pandas_df(['redshift', 'ra', 'dec', 'SFR', 'issb', 'mu','Dlum', 'Umean', 'LIR', 'Mhalo',
                                  'Mstar', 'issb', 'ICO10', 'ICO21', 'ICO32', 'ICO43', 'ICO54', 'ICO65', 'ICO76', 'ICO87', 'ICII_de_Looze', 'ICI10', 'ICI21'])
    #cat =  Table.from_pandas(cat)
    
    end = time.time()
    timing = end - start
    print(f'Loaded in {np.round(timing,2)} sec!')

    cube_gal_params_file = "PAR_FILES/CONCERTO_uchuu_ref_gal_cubes.par"
    cube_params_file = "PAR_FILES/CONCERTO_uchuu_ref_cube.par"
    params_sides_file = 'PAR_FILES/SIDES_from_uchuu.par' 
    
    return 'pySIDES_from_uchuu', cat, cats_dir_path, int(117) #cube_gal_params_file, cube_params_file, params_sides_file, 
    
def worker_init(*args):
    global _args
    _args = args

def worker_compute(params):
    global _args
    cat, simu, field_size, cat_path, line, rest_freq = _args
    for z, dz, n_avg in params: gen_maps(cat, simu, z, n_avg, dz, field_size, cat_path, line, rest_freq)
    return 0

def make_all_cubes(cat, simu, field_size, cat_path,line, rest_freq, ncpus=24):

    tim_params = load_params('PAR/cubes.par')
    z_list = tim_params['z_list']
    dz_list = tim_params['dz_list']
    n_list = tim_params['n_list']
    
    params_list = []
    for z in z_list:
        for n,dz in zip(n_list, dz_list): 
                params_list.append( list((z,dz,n ))) 
                
    print("start parallelization")
    with Pool(ncpus, initializer=worker_init, initargs=list((cat, simu, field_size, cat_path,line, rest_freq ))) as p:
        zero = p.map(worker_compute, np.array_split(params_list, ncpus) )
    return 0   

def gen_maps(cat, simu, 
            z, n_slices, dz, field_size, cat_path, line, rest_freq, mstar_cut = 1e10, 
            gen_continuum=False, gen_galaxies=True, gen_interlopers=True, compute_properties=True, ):
    
    '''
    Generate a map line intensity map as well as a galaxy map with stellar mass cut mstar_cut from a catalogue cat. 
    The corresponding continuum, interlopers, and interlopers-contaminated intensity maps can aslo be generated. 
    The catalog cat is given by simulation like full SIDES-Uchuu, SIDES-Uchuu subfields or SIDES-Bolshoi, covering a field of view field_size. 
    The spectral spatial map is centered on the observed frequency of a given line at redshift z. It covers a spectral width dnu = dz*nu_obs/(1+z). 
    Parameters: 
    cat (pandas dataframe): the catalogue containing the ra,dec,z and emission properties of galaxies. 
    simu (str): the simulation from which the catalogue originates, use to name output files.
    line (str): the line from which nu_obs, the center of the cube, is set. 
    rest_freq (float): the rest_frame frequency of the choosen line [GHz]
    mstar_cut (float): the stellar mass cut to apply to create the galaxy map. 
    continuum (bool): should the continuum map be saved?
    galaxies (bool): should the galaxy map be saved?
    interlopers (bool): should the interlopers map be saved?
    compute_properties (bool): should other properties (such as the shot noise in the map, the background intensity ect...) be computed and saved ?
    '''

    #The prefix used to save the outputs maps:
    params_name = f"{simu}_z{z}_dz{np.round(dz,3)}_{n_slices}slices_{field_size}deg2"
    params_cube = load_params("PAR/cubes.par")
    dkk = params_cube['dkk']
    res = params_cube['pixel_size'] * u.arcsec
    pixel_sr = (res.value * np.pi/180/3600)**2 #solid angle of the pixel in sr
    #Spectral properties of the map:
    nu_obs   = rest_freq / (1+z)
    dnu = dz * nu_obs / (1+z)
    freqs = np.linspace(nu_obs-n_slices*dnu, nu_obs+n_slices*dnu, int(2*n_slices+1))
    nu_min = nu_obs - (n_slices*dnu) 
    nu_max = nu_obs + (n_slices*dnu) 
    nu_min_edge = nu_min - dnu/2
    nu_max_edge = nu_max + dnu/2
    params_cube = {'freq_min':nu_min*1e9, 'freq_max':nu_max*1e9, 'freq_resol':dnu*1e9, "pixel_size":res.value}

    #Select objects emitting the line in [nu_min_edge, nu_max_edge]
    cat_line = cat.loc[ np.abs(  rest_freq/(1+cat['redshift']) - nu_obs) <= (nu_max_edge - nu_min_edge)/2]
    cube_prop_dict = set_wcs(cat_line, params_cube)

    #--- Creates the intrinsic line intensity map and save it ---#
    S, channels = line_channel_flux_densities(line, rest_freq, cat_line, cube_prop_dict)
    line_map, edges = np.histogramdd(sample=(channels, cube_prop_dict['pos'][0], cube_prop_dict['pos'][1]), 
                                     bins =(cube_prop_dict['z_edges'], cube_prop_dict['y_edges'], cube_prop_dict['x_edges']))
    if( line_map.sum() != len(cat_line) ): print('problem A')

    line_map, edges = np.histogramdd(sample=(channels, cube_prop_dict['pos'][0], cube_prop_dict['pos'][1]), 
                                     bins =(cube_prop_dict['z_edges'], cube_prop_dict['y_edges'], cube_prop_dict['x_edges']), weights=S)
    line_map *= 1 / pixel_sr / 1e6
    save_cube(output_path, f"{params_name}", line, "MJy_sr", cube_prop_dict, 'MJy per sr', cat_path, dz, cube=line_map)

    if(gen_galaxies):
        #---Creates the corresponding galaxy map, with a stellar mass cut mstar_cut
        cat_galaxies = cat_line.loc[cat["Mstar"] >= mstar_cut]
        x, y = cube_prop_dict['w'].celestial.wcs_world2pix(cat_galaxies['ra'] , cat_galaxies['dec'], 0)
        freq_obs = rest_freq/(1+cat_galaxies["redshift"])
        channels_gal = np.asarray(cube_prop_dict['w'].swapaxes(0, 2).sub(1).wcs_world2pix(freq_obs*1e9, 0))[0] 
        galaxy_cube, edges = np.histogramdd(sample=(channels_gal, y, x), bins = (cube_prop_dict['z_edges'], cube_prop_dict['y_edges'], cube_prop_dict['x_edges']))
        if( galaxy_cube.sum() != len(cat_galaxies) ): print('problem B')
        save_cube(output_path, f"{params_name}", 'galaxies', 'pix', cube_prop_dict, 'pix', cat_path, dz, cube=galaxy_cube) 

        
    if(gen_continuum):
        lambda_list =  ( cst.c * (u.m/u.s)  / (np.asarray(freqs)*1e9 * u.Hz)  ).to(u.um)
        SED_dict = pickle.load(open('pysides/SEDfiles/SED_finegrid_dict.p', "rb"))
        print("Generate monochromatic fluxes...")
        Snu_arr = gen_Snu_arr(lambda_list.value, SED_dict, cat_line["redshift"], cat_line['mu']*cat_line["LIR"], cat_line["Umean"], cat_line["Dlum"], cat_line["issb"])
        histo, edges = np.histogramdd(sample=(channels,cube_prop_dict['pos'][0], cube_prop_dict['pos'][1]), bins=(cube_prop_dict['z_edges'], cube_prop_dict['y_edges'], cube_prop_dict['x_edges']), weights=Snu_arr[:, 0])
        continuum = histo.value / pixel_sr / 1e6
        save_cube(output_path, f"{params_name}_{line}", 'continuum', "MJy_sr",  cube_prop_dict, 'MJy per sr', cat_path, dz, cube=continuum)

    if(gen_interlopers):
        cube_interlopers = np.zeros(line_map.shape)
        cube_all_lines   = line_map.copy()
        for Jp, rest_freqp in zip(line_list, rest_freq_list):
            if(Jp != line):
                cat_Jline = cat.loc[ np.abs( (rest_freqp.value/(1+cat['redshift']))- nu_obs) <= (nu_max_edge - nu_min_edge)/2]
                if(len(cat_Jline)>0):
                    x, y = cube_prop_dict["w"].celestial.wcs_world2pix(np.asarray(cat_Jline['ra'])*u.deg , np.asarray(cat_Jline['dec'])*u.deg, 0)
                    pos = [y , x]
                    S, channels = line_channel_flux_densities(Jp, rest_freqp, cat_Jline, cube_prop_dict)
                    line_map, edges = np.histogram((channels), bins = cube_prop_dict['z_edges'])
                    if( line_map.sum() != len(cat_Jline) ):
                        print('problem C')
                    line_map, edges = np.histogramdd(sample=(channels, pos[0], pos[1]), bins = (cube_prop_dict['z_edges'],cube_prop_dict['y_edges'], cube_prop_dict['x_edges']), weights=S)
                    line_map *= 1 / pixel_sr / 1e6
                    cube_interlopers += line_map 
                    cube_all_lines += line_map   
                    print(f'This map is interloped by {Jp}')
                    if('CI1' in Jp or 'CI2' in Jp): save_cube(output_path, f"{params_name}_{line}", f'{Jp}_interloper', "MJy_sr", cube_prop_dict, 'MJy per sr', cat_path, dz, cube=line_map)
            if(Jp=='CO87'): 
                save_cube(output_path, f"{params_name}_{line}", 'CO_all', "MJy_sr", cube_prop_dict, 'MJy per sr', cat_path, dz, cube=cube_interlopers)
                save_cube(output_path, f"{params_name}_{line}", 'CO_interlopers', "MJy_sr", cube_prop_dict, 'MJy per sr', cat_path, dz, cube=cube_all_lines)

        save_cube(output_path, f"{params_name}_{line}", 'all_interlopers', "MJy_sr", cube_prop_dict, 'MJy per sr', cat_path, dz, cube=cube_interlopers)
        save_cube(output_path, f"{params_name}_{line}", 'all_lines', "MJy_sr", cube_prop_dict, 'MJy per sr', cat_path, dz, cube=cube_all_lines)

    if(gen_continuum and gen_interlopers):
        full = cube_all_lines + continuum
        save_cube(output_path, f"{params_name}_{line}", 'full', "MJy_sr", cube_prop_dict, 'MJy per sr', cat_path, dz, cube=full)


    if(compute_properties):

        dict_J = compute_other_linear_model_params( f"{params_name}_" + line, f"{params_name}_" + line,
                                                    output_path,line, rest_freq*u.GHz, z, dz, n_slices, field_size, cat)
        
        dict_Jg_noint = powspec_LIMgal(f"{params_name}_" + line, f"{params_name}_" + line, params_name+'_galaxies', output_path,
                                        line,  z, dz, n_slices, field_size, dkk)
        
        dict_Jg_int = powspec_LIMgal(f"{params_name}_"+line, f"{params_name}_"+line+'_all_lines', params_name+'_galaxies', output_path,
                                        line,  z, dz, n_slices, field_size, dkk)
        
        #if(gen_continuum and gen_interlopers):

        #dict_Jg_int = powspec_LIMgal(f"{params_name}_"+line, f"{params_name}_"+line+'_full', params_name+'_galaxies', output_path,
        #                            line,  z, dz, n_slices, field_size, dkk)
        
    return 0

if __name__ == "__main__":

    #parser to choose the simu and where to save the outputs 
    #e.g: python gen_cubes_TIM_cubes_117deg2_uchuu.py 'outputs_uchuu/' 'uchuu'
    #will generate the 117deg2 SIDES-Uchuu maps around z+-dz/2 and saves them in outputs_uchuu


    '''
    parser = argparse.ArgumentParser(description="gen cubes from Uchuu",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('output_path', help="output path of products", default = '.')
    args = parser.parse_args()
    output_path = args.output_path
    '''

    simu, cat, dirpath, fs = load_cat()

    #With SIDES Bolshoi, for rapid tests. 
    '''
    dirpath="/net/CONCERTO/home/mvancuyck/"
    cat = Table.read(dirpath+'pySIDES_from_original.fits')
    cat = cat.to_pandas()
    simu='pySIDES_from_bolshoi'; fs=2
    '''
    params = load_params('PAR/cubes.par')
    params['output_path'] = '/net/CONCERTO/home/mvancuyck/TIM_pysides_user_friendly/OUTPUT_TIM_CUBES_FROM_UCHUU/'
    pars = load_params('PAR/SIDES_from_original_with_fir_lines.par')
    pars['output_path'] = params['output_path']
    for tile_sizeRA, tile_sizeDEC in params['tile_sizes']: 

        if(fs<tile_sizeRA*tile_sizeDEC): continue

        ragrid=np.arange(cat['ra'].min(),cat['ra'].max(),np.sqrt(tile_sizeRA))
        decgrid=np.arange(cat['dec'].min(),cat['dec'].max(),np.sqrt(tile_sizeDEC))
        grid=np.array(np.meshgrid(ragrid,decgrid))
        ra_index = np.arange(0,len(ragrid)-1,1)
        dec_index = np.arange(0,len(decgrid)-1,1)
        ra_grid, dec_grid = np.meshgrid(ra_index, dec_index)
        # Flatten the grids and stack them into a single array
        coords = np.stack((ra_grid.flatten(), dec_grid.flatten()), axis=1)

        for l, (ira, idec) in enumerate(coords):
                
            if l >= params['Nmax']: break 

            cat_subfield=cat.loc[(cat['ra']>=grid[0,idec,ira])&(cat['ra']<grid[0,idec,ira+1])&(cat['dec']>=grid[1,idec,ira])&(cat['dec']<grid[1,idec+1,ira])]
            params['run_name'] = f'{simu}_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg'
            pars['run_name'] = f'{simu}_tile_{l}_{tile_sizeRA}deg_x_{tile_sizeDEC}deg'

            gen_outputs(cat_subfield, pars)

            for J, rest_freq in zip(line_list, rest_freq_list):
                print('')
                #make_all_cubes(cat_subfield, f"{simu}_ntile_{l}", tile_size, dirpath, line=J, rest_freq = rest_freq.value )
                #gen_maps(cat_subfield, f"{simu}_ntile_{l}", 0.64, 0, 0.22, tile_size, dirpath, line=J,rest_freq = rest_freq.value)

    for J, rest_freq in zip(line_list, rest_freq_list):
        print('')
        #make_all_cubes(cat, simu, fs, dirpath, line=J,rest_freq = rest_freq.value )