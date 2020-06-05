#!/usr/bin/env python3

import utils
import inputs
import file_io
import argparse
import interface
import structure
import numpy as np


"""Parse inputs"""
parser = argparse.ArgumentParser(description = "Script for calculating and setting the work of adhesion "\
                                 "from an array of data",\
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                 prog = "setEintArray")

parser.add_argument("-i", "--input", type = str, required = True,\
                    help = "File containing all interfaces")

parser.add_argument("-epa", "--energy_per_area", nargs = 2, type = float, required = True,\
                    help = "Energy per area of the individual surfaces of interest")

parser.add_argument("-v", "--verbose", action = "count", default = 0,\
                    help = "Print extra information")

parser.add_argument("-ai", "--array_input", type = str, required = True,\
                    help = "File containing energy/area data shape(N+1,4), (idx, E_t1, E_t2,...) "\
                    "with a header containing the translations")

opt = parser.parse_args()

"""Load data from array_inputs, should be formatted as shape(N,3) with cols (idx, translation, energy)"""
with open(opt.array_input, 'r') as ai:
    trans = [np.int(i) for i in ai.readline().split()]
    data = np.loadtxt(ai)

"""Load file containing the interfaces"""
itf = utils.loadInterfaces(filename = opt.input, verbose = opt.verbose)

"""Cast index as int"""
index = data[:, 0].astype(np.int) #Index are 0-based

"""Get the area of the specified interfaces"""
#area = itf.getAreas(idx = index, cell = 1)

"""Calculate the energy per area for each interface"""
#energy_per_area = data[:, 1:] / np.tile(area[:, None], (1, np.shape(trans)[0]))

"""Calculate the work of separation E_per_area_interface - E_per_area_W_slab - E_per_area_WC_slab"""
work_sep = data[:, 1:] - opt.energy_per_area[0] - opt.energy_per_area[1]

"""Set the energy"""
itf.setEintArray(idx = index, translation = trans, e_int = work_sep)

"""Save the interfaces"""
itf.saveInterfaces(filename = opt.input, verbose = opt.verbose)
