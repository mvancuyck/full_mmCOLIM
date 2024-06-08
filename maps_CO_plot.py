from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import numpy as np
from matplotlib.colors import LogNorm
import scipy.constants as cst
import astropy.units as u
from astropy.stats import gaussian_fwhm_to_sigma
import astropy.convolution as conv
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython import embed
import random

freq_CII = 1900.53690000 * u.GHz
freq_CI10 = 492.16 *u.GHz
freq_CI21 = 809.34 * u.GHz
rest_freq_list = [115.27120180  *u.GHz* J_up for J_up in range(1, 9)]
rest_freq_list.append(freq_CI10); rest_freq_list.append(freq_CI21); rest_freq_list.append(freq_CII); 
line_list = ["CO{}{}".format(J_up, J_up - 1) for J_up in range(1, 9)]
line_list.append('CI10'); line_list.append('CI21'); line_list.append('CII_de_Looze')

def plot_map( cube_path, J, G, n, BS=8, vmin=1e-6, vmax=1e-2, to_conv=False):
    
    cube_file = f"{cube_path}/{J}"
    gal_file  = f"{cube_path}/{G}"
    hdr= fits.getheader(cube_file)
    nx, ny = hdr["NAXIS2"], hdr["NAXIS1"]
    #n = hdr["NAXIS3"]
    hdr = fits.getheader(cube_file)
    res = (hdr["CDELT1"] * u.Unit(hdr["CUNIT1"])).to(u.rad)
    dnu =(hdr["CDELT3"] * u.Unit(hdr["CUNIT3"]))
    nu_obs= (hdr["CRVAL3"] * u.Unit(hdr["CUNIT3"])) + dnu*n
    wcs = WCS(cube_file)
    x, y,_ = wcs.wcs_pix2world([0, nx], [0, ny],[0,0], 0)
    ra, dec = np.meshgrid(x, y)
    plane = fits.getdata(cube_file)[int(n),:,:]
    planeG = fits.getdata(gal_file)[int(n),:,:]
    if(to_conv):
        #fwhm = 1.22 * cst.c / (nu_obs.to(u.Hz).value * 11.5) * u.rad
        #sigma = (fwhm * gaussian_fwhm_to_sigma).to(u.arcsec)
        #sigma_pix = sigma.value / (res.to(u.arcsec)).value  #pixel
        kernel = conv.Gaussian2DKernel(x_stddev=5, x_size=nx); kernel.normalize(mode="peak")
        planeG = conv.convolve_fft(planeG, kernel, normalize_kernel=True)
    
    plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS);
    fig, ax = plt.subplots(1,1,figsize=(3,3), dpi=200)
    im = ax.imshow(np.abs(plane+1e-10), origin='lower',cmap='Oranges',norm=LogNorm(vmin=vmin, vmax=vmax),extent=(ra.min(), ra.max(), dec.min(), dec.max()) )
    ax.set_xlabel('DEC [$\\rm deg^2$]')
    ax.set_ylabel('RA [$\\rm deg^2$]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, )
    cbar.set_label('Intensity [MJy.$\\rm sr^{-1}$]', rotation=90, labelpad=10)
    fig.tight_layout()
    for extension in ("png", "pdf"): fig.savefig(f"figs/map_with_scale_{J[:-5]}.{extension}", transparent=True,)

    plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS);
    fig, ax = plt.subplots(1,1,figsize=(3,3), dpi=200)
    im = ax.imshow(np.abs(planeG), origin='lower',cmap='gist_gray',vmax = 1, extent=(ra.min(), ra.max(), dec.min(), dec.max()) )
    ax.set_xlabel('DEC [$\\rm deg^2$]')
    ax.set_ylabel('RA [$\\rm deg^2$]')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax, )
    cbar.set_label('Nb of galaxies', rotation=90, labelpad=10)
    fig.tight_layout()
    for extension in ("png", "pdf"): fig.savefig(f"figs/map_with_scale_{G[:-5]}.{extension}", transparent=True,)

    plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS);
    fig, ax = plt.subplots(1,1,figsize=(3,3), dpi=200)
    im = ax.imshow(np.abs(plane+1e-10), origin='lower',cmap='Oranges',norm=LogNorm(vmin=vmin),extent=(ra.min(), ra.max(), dec.min(), dec.max()) )
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    #for extension in ("png", "pdf"): fig.savefig(f"figs/map_{J[:-5]}.{extension}", transparent=True,)

    plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS);
    fig, ax = plt.subplots(1,1,figsize=(3,3), dpi=200)
    im = ax.imshow(np.abs(planeG), origin='lower',cmap='gist_gray', vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    #for extension in ("png", "pdf"): fig.savefig(f"figs/map_{G[:-5]}.{extension}", transparent=True,)

    #if(z==1 and 'all_lines' in J and '32'in J): plt.show()
    plt.close('all')

cube_path = 'outputs_cubes/'
for tile in range(12):
    for z in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        for j in range(8):
            for line in ('', '_all_lines'):#
                J = f'CO{j+1}{j}'
                file_J   = f'pySIDES_from_uchuu_ntile_{int(tile)}_z{z}_dz0.05_2.0slices_9deg2_{J}{line}_MJy_sr.fits'
                file_gal = f'pySIDES_from_uchuu_ntile_{int(tile)}_z{z}_dz0.05_2.0slices_9deg2_galaxies_pix.fits'
                plot_map( cube_path, file_J, file_gal, 2.0)

