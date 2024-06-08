from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import numpy as np
from matplotlib.colors import LogNorm
import scipy.constants as cst
import astropy.units as u
from astropy.stats import gaussian_fwhm_to_sigma
import astropy.convolution as conv
from IPython import embed
from matplotlib.colors import LogNorm

dz = 0.1 

params_list = []
wcs_list    = []
im_list    = []
freq_list = []

trans = 'CO32'

for z in (0.5, 3.0):
    for data in ('', 'all_lines_', 'galaxy'):

        if(data=='galaxy'): 
            type = 'galaxies_pix'
        else: 
            type = f'{trans}_{data}MJy_sr' 

        file = f'outputs_cubes/pySIDES_from_uchuu_ntile_0_z{z}_dz{dz}_0.0slices_9deg2_{type}.fits'
        hdu = fits.open(file)
        wcs = WCS(hdu[0].header) 
        freq = wcs.pixel_to_world(0, 0, 0)[1].to('GHz')
        freq_list.append( freq )
        wcs = WCS(hdu[0].header, naxis=2) 
        wcs_list.append( wcs )
        im = hdu[0].data[0, :, :]
        im_list.append(im) 

        if(data=='galaxy'): 
            title=f'Galaxies at z={z}'
            cmap='gist_gray'
            vmin=0
            vmax=10
        else: 
            if('all' in data): title = f'{trans} and interlopers \n at (z={z},'+' $\\rm \\nu$={:0.0f})'.format(freq)
            else: title = f'{trans} at z={z}'
            cmap = 'cividis'
            vmax = 0.008*1e3
            vmin = 0.1
        params_list.append( list((data,z, cmap, title, vmin, vmax)))


BS=10; plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS); lw=1; mk=5; elw=1
fig, axs = plt.subplots(2, 3, #sharex=True, sharey = 'row', #gridspec_kw={'height_ratios': [1,1]}, 
                        figsize=(11,8), dpi = 200,
                        subplot_kw={'projection': wcs_list[0]} )
axs = np.ravel(axs)

for iax, (data, z, cmap, title, vmin, vmax) in enumerate(params_list):
    wcs = wcs_list[iax]
    freq = freq_list[iax]
    if('Gal' in title): im = axs[iax].imshow(im_list[iax], origin='lower', vmax=1, cmap=cmap,)
    else: 
        im = axs[iax].imshow(1.e3*im_list[iax], origin='lower', cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))#, projection=wcs)    
        if(iax==1): im1 = im
    axs[iax].set_xlabel('RA')
    #if(iax==0 or iax==3): 
    axs[iax].set_ylabel('Dec')
    axs[iax].set_title(title)

# Create colorbar for cividis colormap
cbar_ax_cividis = fig.add_axes([0.92, 0.55, 0.02, 0.35])  # [left, bottom, width, height]
cbar_cividis = fig.colorbar(im1, cax=cbar_ax_cividis)
cbar_cividis.set_label('$\\rm B_{\\nu}$ [kJy/sr]')

# Create colorbar for Greys colormap
cbar_ax_greys = fig.add_axes([0.92, 0.15, 0.02, 0.35])  # [left, bottom, width, height]
cbar_greys = fig.colorbar(im, cax=cbar_ax_greys)
cbar_greys.set_label('Galaxies count')
cbar_greys.set_ticks([0, 1])

fig.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust the layout to make space for the colorbars
fig.savefig(f'figs/Slice_{freq}GHz.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
        
