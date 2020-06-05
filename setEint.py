#!/usr/bin/env python3

import utils
import inputs
import file_io
import argparse
import interface
import structure
import numpy as np


"""Parse inputs"""
parser = argparse.ArgumentParser(description = "Script for calculating and setting the work of adhesion",\
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                 prog = "setEint")

parser.add_argument("-i", "--input", type = str, required = True,\
                    help = "File containing all interfaces")

parser.add_argument("-idx", "--index", type = int, required = True,\
                    help = "Index of the interface")

parser.add_argument("-epa", "--energy_per_area", nargs = 2, type = float, required = True,\
                    help = "Energy per area of the individual surfaces of interest")

parser.add_argument("-etpa", "--energy_total_per_area", type = float, required = True,\
                    help = "Total energy per area of the interface combination")

parser.add_argument("-t", "--translation", type = int, required = True,\
                    help = "Index of the used translation (starts at 1...)")

parser.add_argument("-v", "--verbose", action = "count", default = 0,\
                    help = "Print extra information")

opt = parser.parse_args()

"""Load the interfaces from predefined .pkl file"""
itf = utils.loadInterfaces(filename = opt.input)

"""Get the area of the interface (cell_1)"""
#area_cell_1 = itf.getArea(idx = opt.index, cell = 1)

"""Calculate the total energy / area (cell_1) for the complete interface"""
#energy_interface = opt.energy_total / area_cell_1

"""Calculate the work of separation E/A (interface) - E/A (slab 1) - E/A (slab 2)"""
work_sep = opt.energy_total_per_area - opt.energy_per_area[0] - opt.energy_per_area[1]

"""Set the value and save"""
itf.setEint(idx = opt.index, e_int = work_sep, translation = opt.translation, verbose = opt.verbose)
itf.saveInterfaces(filename = opt.input, verbose = opt.verbose)
