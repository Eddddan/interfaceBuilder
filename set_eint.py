#!/usr/bin/env python3

import utils
import inputs
import file_io
import argparse
import interface
import structure
import numpy as np


"""Parse inputs"""
parser = argparse.ArgumentParser(description = "Script for setting interfacial energies",\
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                 prog = "set_eint")

parser.add_argument("-i", "--input", type = str, required = True,\
                    help = "File containing all interfaces")

parser.add_argument("-idx", "--index", type = int, required = True,\
                    help = "Index of the interface")

parser.add_argument("-epa", "--energy_per_area", nargs = 2, type = float, default = [0, 0],\
                    help = "Energy per area of the individual bulk configurations of interest")

parser.add_argument("-et", "--energy_total", type = float, required = True,\
                    help = "Total energy per area of the interface combination")

parser.add_argument("-t", "--translation", type = int, required = True,\
                    help = "Index of the used translation (0-based)")

parser.add_argument("-v", "--verbose", action = "count", default = 0,\
                    help = "Print extra information")

opt = parser.parse_args()

"""Load the interfaces from predefined .pkl file"""
itf = utils.loadInterfaces(filename = opt.input)

"""Calculate the work of separation E/A (interface) - E/A (slab 1) - E/A (slab 2)"""
e_int = opt.energy_total - opt.energy_per_area[0] - opt.energy_per_area[1]

"""Set the value and save"""
itf.setEint(idx = opt.index, e_int = e_int, translation = opt.translation, verbose = opt.verbose)
itf.saveInterfaces(filename = opt.input, verbose = opt.verbose)
