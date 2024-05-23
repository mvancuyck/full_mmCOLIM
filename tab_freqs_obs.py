
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
rest_freq_list = [115.27120180 J_up for J_up in range(1, 9)]
dz=0.1


print(r'\begin{table}[h!]')
print(r'\centering')
print(r'\caption{Table of observed frequencies (in GHz) for each transition \jup\,. \label{tab:freq_obs}}')
print(r'\small{\begin{tabular}{p{0.1\linewidth}ccccccc}')
print(r'\hline')
print(r'\hline')
print(r'\noalign{\smallskip}')
L = 'Line & $\rm \nu_{rf}$ '
Lp = ' & '
for z in (0.5, 1.0, 1.5, 2.0,2.5,3.0):
    L+=' & $\rm \nu_{obs}$'
    Lp+= f' & z={z}'
print(L+r'\\')
print(Lp+r'\\')
print(r'\hline')
print(r'\noalign{\smallskip}')
for line, rest_freq in zip(line_list_fancy, rest_freq):
    L = line
    for z in (0.0, 0.5, 1.0, 1.5, 2.0,2.5,3.0):
        nu_obs = rest_freq/(1+z)
        L+= f' & {np.round(nu_obs, 1)}'
    print(L+r'\\')
    print(r'\noalign{\smallskip}')
print(r'\hline')
print(r'\end{tabular}}')
print(r'\end{table}')

print(r'\begin{table}[h!]')
print(r'\centering')
print(r'\caption{Table of corresponding frequency interval (in GHz) to d$z$={dz} for each transition \jup\,.}')
print(r'\small{\begin{tabular}{p{0.1\linewidth}cccccc}')
print(r'\hline')
print(r'\hline')
print(r'\noalign{\smallskip}')
L = 'Line & '
Lp = ' & '
for z in (0.5, 1.0, 1.5, 2.0,2.5,3.0):
    L+=' & $\rm d\nu$'
    Lp+= f' & z={z}'
print(L+r'\\')
print(Lp+r'\\')
print(r'\hline')
print(r'\noalign{\smallskip}')
for line, rest_freq in zip(line_list_fancy, rest_freq):
    L = line
    for z in (0.5, 1.0, 1.5, 2.0,2.5,3.0):
        nu_obs = rest_freq/(1+z)
        dnu = dz* nu_obs / 1+z
        L+= f' & {np.round(dz, 1)}'
    print(L+r'\\')
    print(r'\noalign{\smallskip}')
print(r'\hline')
print(r'\end{tabular}}')
print(r'\end{table}')
