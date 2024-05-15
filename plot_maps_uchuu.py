from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import numpy as np
from matplotlib.colors import LogNorm
import scipy.constants as cst
import astropy.units as u
from astropy.stats import gaussian_fwhm_to_sigma
import astropy.convolution as conv

z=1.0; dz=0.05; n=0.0; line='CO32'; dtype = ''; tile=11; vmin=1e-4
cube_path = '/home/mvancuyck/sides/PYSIDES/OUTPUT_mathilde/original_cat/copaper/'
cube_name = f'pySIDES_from_uchuu_tile_{tile}_z{z}_dz{dz}_{n}slices_9deg2_{line}{dtype}_MJy_sr.fits'

def plot_map( cube_path, cube_file, n, BS=12, vmin=1e-6, to_conv=True):
    
    cube_file = f"{cube_path}/{cube_name}"
    wcs = WCS(cube_file)
    x, y,_ = wcs.wcs_pix2world([0, nx], [0, ny],[0,0], 0)
    ra, dec = np.meshgrid(x, y)
    nx, ny = hdr["NAXIS2"], hdr["NAXIS1"]
    hdr = fits.getheader(cube_file)
    res = (hdr["CDELT1"] * u.Unit(hdr["CUNIT1"])).to(u.rad)
    nu_obs= (hdr["CRVAL3"] * u.Unit(hdr["CUNIT3"])) + (hdr["CDELT3"] * u.Unit(hdr["CUNIT3"]))*n

    fwhm = 1.22 * cst.c / (nu_obs.to(u.Hz).value * 11.5) * u.rad
    sigma = (fwhm * gaussian_fwhm_to_sigma).to(u.arcsec)
    sigma_pix = sigma.value / (res.to(u.arcsec)).value  #pixel
    print(res.to(u.arcsec), fwhm.to(u.arcsec), nu_obs.to(u.GHz))
    kernel = conv.Gaussian2DKernel(x_stddev=sigma_pix, x_size=nx); kernel.normalize(mode="peak")

    plane = fits.getdata(cube_file)[int(n),:,:]
    if(to_conv): plane = conv.convolve_fft(plane, kernel, normalize_kernel=True)
    
    plt.rc('font', size=BS); plt.rc('axes', titlesize=BS); plt.rc('axes', labelsize=BS);
    fig, ax = plt.subplots(1,1,figsize=(6,6), dpi=200)
    im = ax.imshow(np.abs(plane), origin='lower',cmap='Oranges',norm=LogNorm(vmin=vmin),extent=(ra.min(), ra.max(), dec.min(), dec.max()) )
    ax.set_xlabel('DEC [$\\rm deg^2$]')
    ax.set_ylabel('RA [$\\rm deg^2$]')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    cb_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('Intensity [MJy.$\\rm sr^{-1}$]')  # Add the label to the color bar
    for extension in ("png", "pdf"): plt.savefig(f"map_with_scale_{cube_name[:-5]}.{extension}")

    fig, ax = plt.subplots(1,1,figsize=(2,2), dpi=200)
    im = ax.imshow(np.abs(plane), origin='lower',cmap='Oranges',norm=LogNorm(vmin=vmin))
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    for extension in ("png", "pdf"): plt.savefig(f"map_noscale_{cube_name[:-5]}.{extension}")
    plt.show()
