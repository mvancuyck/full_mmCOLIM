import pandas as pd
from IPython import embed
import numpy as np
from astropy.cosmology import Planck15 as cosmo
from time import time
from astropy import units as u
import pickle

def gen_CO(cat, params):

    tstart = time()

    if not ("Dlum" in cat.columns):
        print("Compute luminosity distances because it was not done previously...")
        cat = cat.assign(Dlum = cosmo.luminosity_distance(cat["redshift"])) #Mpc

    print('Compute the CO(1-0) fluxes...')

    if "LIR" in cat.columns:
        LIR = cat['LIR']
    else:
        LIR = cat['SFR'] * params['SFR2LIR']  
    #recipe based on Sargent et al. 2014 to generate CO(1-0)

    LprimCO10 = np.zeros(len(cat))
    sel = np.where(cat['qflag'] == False)
    LprimCO10[sel[0]]= 10.**( 0.81 * np.log10(LIR[sel[0]]) + 0.54 ) # in K km/s pc^2, the +10. after SFR is for Kennicutt conversion (Chabrier IMF)

    sb = np.where(cat['issb'])
    
    LprimCO10[sb[0]] = LprimCO10[sb[0]] * 10.**(-0.46)

    cat = cat.assign(LprimCO10 = LprimCO10)

    #add scatter on CO1-0 luminosity
    
    cat['LprimCO10'] *=  10**(params['sigma_dex_CO10'] * np.random.normal(size = np.size(cat['LprimCO10'])))

    nu_CO_obs = params['nu_CO'] / (1. + cat["redshift"])

    cat = cat.assign( ICO10 = cat['LprimCO10'] * (1. + cat['redshift'])**3 * nu_CO_obs ** 2 / cat['Dlum']**2 / 3.25e7 )
    
    #Daddi et al. (2015) to generate the SLEDS
   
    Jup, Idiffuse, Iclump = np.loadtxt(params['SLED_filename'], unpack = True, comments = '#')

    #normalize the SLED to 1 for the 1-0 tansition
    Idiffuse = Idiffuse / Idiffuse[0]
    Iclump = Iclump / Iclump[0]

    #Daddi et al. 2015: log(ICO54/ICO21) = 0.6 * log(<U>) - 0.38

    R54_21 = 10.**( 0.6 * np.log10(cat['Umean']) -0.38)

    fclump = ( Idiffuse[4] - R54_21 * Idiffuse[1] ) / ( R54_21 * (Iclump[1] - Idiffuse[1]) - (Iclump[4] - Idiffuse[4]))

    neg = np.where(fclump < 0)
    if np.size(neg) > 0:
        fclump[neg[0]] = 0.

    sup1 = np.where(fclump > 1)
    if np.size(sup1) > 0:
        fclump[sup1[0]] = 1.


    #Allow to use an alternative SLED for the starburst else use the same recipe for all the objects (from Birkin, Weiss, Wardlow et al. 2020)

    rJup1_Birkin = np.array([0.9,0.6,0.32,0.35,0.3,0.22,0.22*(7/8)**2]) # be careful, it is a L' ratio and not a line flux ratio!, assume the same flux ratio for 8-7 and 7-6 transition as suggested by their figure 5, starts with r21

    for k in range(1,8):
        print('Work on the ICO{:d}{:d} lines'.format(k+1,k))
    
        if params['SLED_SB_Birkin'] == True:
            Ivec = np.zeros(len(cat))
            
            ms = np.where((cat['issb'] == False) & (cat['qflag'] == False))
            Ivec[ms[0]] = (fclump[ms[0]] * Iclump[k] + (1 - fclump[ms[0]]) * Idiffuse[k])
            
            sb = np.where(cat['issb'] == True)
            Ivec[sb[0]] = rJup1_Birkin[k-1] * (k+1)**2 #the (k+1)**2 square comes from the nu_rest**2 factor in the equation linking L' and I. 
            
            Ivec *= cat['ICO10']
        else:
            Ivec = cat['ICO10'] * (fclump * Iclump[k] + (1 - fclump) * Idiffuse[k])
            
        kwargs = {'ICO{:d}{:d}'.format(k+1,k):Ivec} 
        cat = cat.assign(**kwargs)

    tstop = time()

    print('CO line fluxes of ', len(cat), ' galaxies generated in ', tstop-tstart, 's')

    return cat

    ######[CII]#####

def gen_CII(cat, params):

    tstart = time()

    if not ("Dlum" in cat.columns):
        print("Compute luminosity distances because it was not done previously...")
        cat = cat.assign(Dlum = cosmo.luminosity_distance(cat["redshift"])) #Mpc

    nu_CII_obs = params['nu_CII'] / (1. + cat["redshift"])

    #Lagache & Cousin relation
    if params['generate_Lagache'] == True:
        print('Compute the [CII] fluxes using the Lagache relation....')
        cat = cat.assign( LCII_Lagache = cat["SFR"]**(1.4 - 0.07 * cat["redshift"]) * 10**( 7.1 - 0.07 * cat["redshift"]) )
        #add scatter
        cat['LCII_Lagache'] *=  10**(params['sigma_dex_CII'] * np.random.normal(size = np.size(cat['LCII_Lagache'])))

        cat = cat.assign(ICII_Lagache = cat['LCII_Lagache'] / 1.04e-3 / cat['Dlum']**2 / nu_CII_obs)

    #De Looze relation (HII/starburst galaxies, linear in this case)
    if params['generate_de_Looze'] == True:
        print('Compute the [CII] fluxes using the de Looze relation....')
        cat = cat.assign( LCII_de_Looze =  10.**7.06 * cat['SFR'])
        #add scatter
        cat['LCII_de_Looze'] *=  10**(params['sigma_dex_CII'] * np.random.normal(size = np.size(cat['LCII_de_Looze'])))

        cat = cat.assign(ICII_de_Looze =  cat['LCII_de_Looze'] / 1.04e-3 / cat['Dlum']**2 / nu_CII_obs)

    tstop = time()

    print('[CII] line fluxes of ', len(cat), ' galaxies generated in ', tstop-tstart, 's')

    return cat

    ######[CI]#####

def gen_CI(cat, params):

    tstart = time()

    if not ("Dlum" in cat.columns):
        print("Compute luminosity distances because it was not done previously...")
        cat = cat.assign(Dlum = cosmo.luminosity_distance(cat["redshift"])) #Mpc

    if not ( ("ICO43" in cat.columns) or ("ICO76" in cat.columns)):
        print("WARNING!!!!! CO fluxes must be generated before the [CI] fluxes, since they are derived from them! No CI flux generated!!!!!!")
        return cat

    #Use the empirical relations calibrated in Calibrated_CI_recipe.ipynb in LINEPLOT (based on Valentino+20 sample)
    nu_CO43 = 4 * params['nu_CO']
    ICI10 = np.zeros(len(cat))
    ICI21 = np.zeros(len(cat))

    if "LIR" in cat.columns:
        LIR = cat['LIR']
    else:
        LIR = cat['SFR'] * params['SFR2LIR']
                 
    #compute only the values for the sources with LIR>0. Else, flux is 0!
    sel = np.where(LIR > 0)
    
    logLCO43_LIR_sf = np.log10( 1.04e-3 * cat['ICO43'][sel[0]] * cat['Dlum'][sel[0]]**2 * nu_CO43 / (1 + cat['redshift'][sel[0]]) / cat['LIR'][sel[0]] )
    LCI10_sf = 10.**(params['a_CI10'] * logLCO43_LIR_sf + params['b_CI10']) * LIR[sel[0]] * 10.**(params['sigma_CI10'] * np.random.normal(size = len(sel[0])))
    ICI10[sel[0]] =  LCI10_sf * (1 + cat['redshift'][sel[0]]) / (1.04e-3 * cat['Dlum'][sel[0]]**2 * params['nu_CI10'])
  

    logLCO76_LCO43_sf = np.log10( cat['ICO76'][sel[0]] / cat['ICO43'][sel[0]]) + np.log10(7./4.) #the second term is coming from the ratios of the nu_obs when going from luminosity in Lsun to flux in Jy km/s
    LCI21_sf = 10.**(params['a_CI21'] * logLCO76_LCO43_sf + params['b_CI21']) * LCI10_sf * 10.**(params['sigma_CI21'] * np.random.normal(size = len(sel[0])))

    ICI21[sel[0]] =  LCI21_sf * (1 + cat['redshift'][sel[0]]) / (1.04e-3 * cat['Dlum'][sel[0]]**2 * params['nu_CI21'])

    cat = cat.assign(ICI10 = ICI10) ##!!
    cat = cat.assign(ICI21 = ICI21) ##!!

    tstop = time()

    print('[CI] line fluxes of ', len(cat), ' galaxies generated in ', tstop-tstart, 's')

    return cat

    ######Far-IR lines, starts with the most obvious ones to test#####

def gen_fir_lines(cat, params):

    tstart = time()

    if not ("Dlum" in cat.columns):
        print("Compute luminosity distances because it was not done previously...")
        cat = cat.assign(Dlum = cosmo.luminosity_distance(cat["redshift"])) #Mpc

    if "LIR" in cat.columns:
        LIR = cat['LIR']
    else:
        LIR = cat['SFR'] * params['SFR2LIR'] 

    sel = np.where(LIR > 0)

    if params['fir_lines_method'] == 'Spinoglio2012':

        #List of the lines and parameters in the paper (PAH excluded), starburst recipe
        line_list =  ['NeII13', 'NeIII16', 'H2_17', 'SIII19', 'OIV26', 'SIII33', 'SiII35', 'OIII52', 'NIII57', 'OI63', 'OIII88', 'NII122','OI145']
        line_slope = [1.17,1.33,1.28,1.09,1.24,1.09,1.11,0.88,0.78,0.98,0.98,1.01, 0.89]
        line_norm = [-3.94,-5.19,-5.42,-4.07,-5.45,-3.63,-3.55,-2.76,-2.78,-2.95,-3.11,-3.80,-3.78]
        
        #all luminosities in Spiniglio are in 10^41 erg/s
        LIR_S12_units = (np.array(LIR) * u.Lsun).to('erg/s').value * 1.e-41
        
        for line, slope, norm in zip(line_list, line_slope, line_norm):
            #print('Generate '+line+'...')

            if (line in params['fir_lines_list']):
                index_nu = params['fir_lines_list'].index(line)
                nu = params['fir_lines_nu'][index_nu]
                Sline = np.zeros(len(cat))
                Lline_sf = 10.**( slope * np.log10(LIR_S12_units[sel[0]]) + norm + (41 - 7) - np.log10(params['Lsun2W'])) #Final result in Lsun    
                Lline_sf *=  10**(params['sigma_dex_fir_lines'] * np.random.normal(size = np.size(Lline_sf)))
                Sline[sel[0]] = Lline_sf * (1 + cat['redshift'][sel[0]]) / (1.04e-3 * cat['Dlum'][sel[0]]**2 * nu) #*cat['mu'][sel[0]] ##!!
                kwargs = {'I'+line:Sline}
                cat = cat.assign(**kwargs) 
            else:
                print(line+' is not listed in the parameter file for the fir_lines_list keyword, and it will thus be skipped!')
    

    line = 'NII205' #Empirical relation from Zhao et al. 2013. 
    
    #print('Generate '+line+'...')
    slope = 1.01
    norm = - 4.25
    index_nu = params['fir_lines_list'].index(line)
    nu = params['fir_lines_nu'][index_nu]
    L_NII205 = 10.**( slope * np.log10(LIR[sel[0]]) + norm ) #Final result in Lsun 
    L_NII205 *=  10**(params['sigma_dex_fir_lines'] * np.random.normal(size = np.size(L_NII205)))
    Sline = np.zeros(len(cat))
    Sline[sel[0]] = L_NII205 * (1 + cat['redshift'][sel[0]]) / (1.04e-3 * cat['Dlum'][sel[0]]**2 * nu) #*cat['mu'][sel[0]] ##!!
    kwargs = {'I'+line:Sline}  
    cat = cat.assign(**kwargs)  

    tstop = time()
    print('Far-IR line fluxes of ', len(cat), ' galaxies generated in ', tstop-tstart, 's')

    return cat
            

        

def make_log_bin(min_logbin_value, max_logbin_value, nbins):
    log_bins = np.linspace(min_logbin_value, max_logbin_value, nbins)
    Deltabin = (max_logbin_value - min_logbin_value) * 1. / nbins #in dex
    log_mean = log_bins[:-1] + Deltabin / 2
    return log_bins, Deltabin, log_mean
        

