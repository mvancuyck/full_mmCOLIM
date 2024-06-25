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

catfir = gen_fir_lines(cat, params_sides)

gen_outputs(catfir, params_sides)