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


    colors_co = ('orange', 'r', 'b', 'cyan', 'g', 'purple', 'magenta', 'grey',)
    b_list, I_list = bI_from_autospec_and_cat(n_slices,9,0.15, z_list, dz_list )
    bI_tinker = bI_from_tinker(n_slices,9,0.15, z_list, dz_list)
    bI_mes, bI_mesCO76, bI_mesCI = bI_crosspec(n_slices,9,0.15, z_list, dz_list, b_list,dtype = '_with_interlopers')
    idz=0
    
    patchs=[]
    BS = 8; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(5,5), dpi=200); 
    mks=6; dr = 0.04; dy = 0.0; text_y_offset=0.2; lw=1; width = [1,1];  height = [1,1,1,1,1]
    drelative = (-4,-3,-2,-1,0,1,2,3)
    gs = gridspec.GridSpec(ncols=2, nrows=5, height_ratios= height, width_ratios=width)
    axr = plt.subplot(gs[-2]); patchs = []
    axr.set_xlabel(r"redshift")
    axr.set_ylabel("relative \n difference")
    axr.set_ylim(-0.3, 1.2)
    for j, (line, rest_freq) in enumerate(zip(line_list_fancy[:8], rest_freq_list[:8])):
        c=colors_co[j]; 
        #---------------------------
        ax = plt.subplot(gs[j]); 
        #---------------------------
        if(j==7): patch = mpatches.Patch(color='grey', label=r'from cross-power spectrum estimate' ); patchs.append(patch)
        ax.errorbar(    z_list, np.mean(bI_mes[:,:,idz,j], axis=0),  linestyle='solid', color=c, ecolor=c, lw=lw) 
        ax.fill_between(z_list, np.mean(bI_mes[:,:,idz,j], axis=0)-np.std(bI_mes[:,:,idz,j], axis=0), np.mean(bI_mes[:,:,idz,j], axis=0)+np.std(bI_mes[:,:,idz,j], axis=0), alpha = 0.2, color=c)
        if(j==7): patch = mlines.Line2D([], [], color='k', linestyle=":", marker="*",  label='catalogue mean brightness \ntimes effective bias'); patchs.append(patch)
        ax.errorbar( z_list, np.mean(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0), yerr=np.std(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0),
                     linestyle=":", color='k', ecolor='k', marker='*',markersize = mks,lw=lw )
        if(j==7): patch = mlines.Line2D([], [], color='k', linestyle="dashdot", marker="p", mfc='none', label=r'Tinker et al. 2010'); patchs.append(patch)
        ax.errorbar( z_list, np.mean(bI_tinker[:, :, idz, j], axis=0), yerr=np.std(bI_tinker[:, :, idz, j], axis=0),
                     linestyle="dashdot", color='k', ecolor='k', marker="p",markersize = mks,lw=lw,  mfc='none', mec='k')
        if(j==7): ax.legend(handles = patchs, loc='center left', bbox_to_anchor=(-0.4, -0.7), fontsize=7, frameon=False)
        axr.errorbar( np.asarray(z_list)+drelative[j]*dr, (np.mean(bI_mes[:,:,idz,j], axis=0)/np.mean(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0))-1+dy, 
                     yerr = (np.std(bI_mes[:,:,idz,j], axis=0)/np.mean(b_list[:, :, idz, j,0,0]*I_list[:,:,idz,j], axis=0)), linestyle="None", color=c,  ecolor=c, marker='*', markersize = mks, lw=lw )
        #print(f'Mean uncertainty of J={j+1}:'+f'{100*np.mean( sigma_bI[:,j,0]/bI[:,j,0] )}'+'%')
        #print(f'Mean diff with intrinsec of J={j+1}:'+f'{100*np.mean( bI[:,j,0]/bIcons[:,j,0]-1)}'+'%')
        #print('')
        #---------------------------
        ax.set_ylabel(r"$b_{\rm eff} \times$"+"$\\rm B^{"+'{}'.format(line)+"}_{\\nu}$ ")
        ax.set_xlim(0.4, np.max(z_list)+0.1)
        ax.set_yscale("log")
        if(j!=7): ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=False)
        else:
            ax.tick_params(axis = "x", which='major', tickdir = "inout", bottom = True, top = True, labeltop=False,labelbottom=True)
        if(j == 7): ax.set_xlabel(r"redshift")
    fig.tight_layout(); fig.subplots_adjust(hspace=.0)
    #---------------------------
    for extension in ("png", "pdf"): plt.savefig(f"bI_CO_nslice{nslice}_dz{dz_list[0]}.{extension}", transparent=True)
    plt.show()

    BS = 11; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS)
    fig = plt.figure(figsize=(3,4.5), dpi=200); lw=1; mks=6; j=6; c=colors_co[6]

    plt.errorbar(    z_list, np.mean(bI_mes[:,:,idz,-2], axis=0),  linestyle='solid', color=colors_co[-2], ecolor=c, lw=lw,  label="CO(7-6) cross-power,\n with interlopers",) 
    plt.fill_between(z_list, np.mean(bI_mes[:,:,idz,-2], axis=0)-np.std(bI_mes[:,:,idz,-2], axis=0), np.mean(bI_mes[:,:,idz,-2], axis=0)+np.std(bI_mes[:,:,idz,-2], axis=0), alpha = 0.2, color=colors_co[-2])
    plt.errorbar(    z_list, np.mean(b_list[:, :, idz, -2,0,0]*I_list[:,:,idz,-2], axis=0), yerr=np.std(b_list[:, :, idz, -2,0,0]*I_list[:,:,idz,-2], axis=0), 
                 label = 'catalogue mean brightness \ntimes fitted effective bias',
                 linestyle=":", color='k', ecolor='k', marker='*',markersize = mks,lw=lw )
    c='grey'
    plt.errorbar(  z_list, np.mean(bI_mesCO76[:,:, idz],axis=0), linestyle='--', color=c, ecolor=c,  label='CO(7-6) cross-power,\n without interlopers', lw=lw) 
    plt.fill_between(z_list, np.mean(bI_mesCO76[:,:, idz],axis=0) - np.std(bI_mesCO76[:,:, idz],axis=0), np.mean(bI_mesCO76[:,:, idz],axis=0) +  np.std(bI_mesCO76[:,:, idz],axis=0), alpha = 0.2, color=c,)
    c='purple'
    plt.errorbar( z_list, np.mean(bI_mesCI[:,:, idz],axis=0),  linestyle='--', color=c, ecolor=c,  label='CI(2-1) Cross-power', lw=lw) # yerr = sigma_bI_all_subfiels[:,j,0],
    plt.fill_between(z_list, np.mean(bI_mesCI[:,:, idz],axis=0) - np.std(bI_mesCI[:,:, idz],axis=0), np.mean(bI_mesCI[:,:, idz],axis=0) +  np.std(bI_mesCI[:,:, idz],axis=0), alpha = 0.2, color=c,)
    
    #plt.errorbar(z_list, mean_cross, yerr = sigma_sum_cross, ls='--',c='dimgray',lw=lw, label='(CO76+[CI]21) cross-power)')    
    line='CO(7-6)'
    plt.ylabel(r"$b_{\rm eff} \times$"+"$\\rm B^{"+'{}'.format(line)+"}_{\\nu}$ ")
    plt.legend(fontsize=6, loc='lower right' , frameon=False)
    plt.xlim(0.4, np.max(z_list)+0.1); 
    plt.yscale("log")
    plt.xlabel(r"redshift")
    plt.tight_layout()
    for extension in ("png", "pdf"): plt.savefig(f"CO76.{extension}") 
    
tim_params = load_params('PAR/cubes.par')
z_list = tim_params['z_list']
dz_list = tim_params['dz_list']
n_list = tim_params['n_list']

for i, (dz, nslice) in enumerate(zip(dz_list, n_list)): 
    CO10_LF(z_list, 0.1)
