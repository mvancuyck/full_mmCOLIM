from pysides.load_params import *
from pysides.gen_outputs import *
from pysides.gen_lines import *
from IPython import embed
from pysides.make_cube import *
from gen_all_sizes_cubes import load_cat
recompute = False 

params_sides = load_params('PAR/SIDES_from_original_with_fir_lines.par', force_pysides_path = '.')

simu, cat, dirpath, fs = load_cat()

params_sides['output_path'] = '.'

Nmax=200; 
for tile_size in (0.2,1,2,9):
    if(fs<tile_size): continue

    ragrid=np.arange(cat['ra'].min(),cat['ra'].max(),np.sqrt(tile_size))
    decgrid=np.arange(cat['dec'].min(),cat['dec'].max(),np.sqrt(tile_size))
    grid=np.array(np.meshgrid(ragrid,decgrid))

    ra_index = np.arange(0,len(ragrid)-1,1)
    dec_index = np.arange(0,len(decgrid)-1,1)
    ra_grid, dec_grid = np.meshgrid(ra_index, dec_index)
    # Flatten the grids and stack them into a single array
    coords = np.stack((ra_grid.flatten(), dec_grid.flatten()), axis=1)

    for l, (ira, idec) in enumerate(coords):
        
        if l+1 > Nmax: break 
        cat_subfield=cat.loc[(cat['ra']>=grid[0,idec,ira])&(cat['ra']<grid[0,idec,ira+1])&(cat['dec']>=grid[1,idec,ira])&(cat['dec']<grid[1,idec+1,ira])]
        cat_subfield = gen_fir_lines(cat_subfield, params_sides)


        dz=0.1
        z_list = np.arange(0.5,3.0, dz)
        log_L_bins, L_Deltabin, log_L_mean = make_log_bin(5, 12, 20)
        dict = {'log L #solar lum':log_L_mean, 'z':z_list, 'dz':dz}

        line_list =  ['NeII13', 'NeIII16', 'H2_17', 'SIII19', 'OIV26', 'SIII33', 'SiII35', 'OIII52', 'NIII57', 'OI63', 'OIII88', 'NII122','OI145', 'NII205']

        for line in line_list:
            LF_list = np.asarray((len(z_list), np.min((200, len(coords))), len(log_L_mean) ))
            index_nu = params_sides['fir_lines_list'].index(line)
            nu = params_sides['fir_lines_nu'][index_nu]
            for iz, z in enumerate(z_list): 
                subcat = cat_subfield.loc[ (cat['redshift']>z-dz/2) and (cat_subfield['redshift']<=z+dz/2)]
                logL_inzbin = np.log10(subcat['I'+line] * (1.04e-3 * subcat['Dlum']**2 * nu / (1 + subcat['redshift']))) 
                Vslice = (tile_size*u.deg**2).to(u.sr) / 3 * (cosmo.comoving_distance(z+dz/2)**3-cosmo.comoving_distance(z-dz/2)**3)
                histo = np.histogram(logL_inzbin, bins = log_L_bins, range = (5, 12))
                LF_list[iz, l, :]= histo[0] / L_Deltabin / Vslice
    
        dict[f'LF_{line} #solar lum per Mpc3'] = LF_list
        pickle.dump(dict, open(f'FIR_lines_LFs_of_SIDES_Uchuu_{tile_size}deg2.p', 'wb'))

