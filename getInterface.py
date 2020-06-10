#!/usr/bin/env python3

import utils
import inputs
import file_io
import argparse
import interface
import structure
import numpy as np


"""Parse inputs"""
parser = argparse.ArgumentParser(description = "Script for retreving interfaces",\
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                 prog = "getInterface")

parser.add_argument("-i", "--input", type = str, required = True,\
                    help = "File containing all interfaces")

parser.add_argument("-idx", "--index", type = int, required = True,\
                    help = "Index of the interface to be built")

bs = ["0001", "10-10"]
parser.add_argument("-bs", "--bottom_surface", required = True, choices = bs,\
                    help = "Index of bottom surface")

parser.add_argument("-t", "--translate", nargs = "+", default = [-1], type = int,\
                    help = "Index of translation applied to the interface, 0-based")

parser.add_argument("-d", "--distance", type = float, default = 2.0,\
                    help = "Distance between the surfaces")

parser.add_argument("-vac", "--vacuum", type = float, default = 10.0,\
                    help = "Vacuum added above the interface")

parser.add_argument("-v", "--verbose", action = "count", default = 0,\
                    help = "Print extra information")

parser.add_argument("-zb", "--repeat_bottom", type = int, default = 3,\
                    help = "Repetitions in the Z direction for the bottom surface")

parser.add_argument("-zt", "--repeat_top", type = int, default = 3,\
                    help = "Repetitions in the Z direction for the top surface")

opt = parser.parse_args()

"""Load the interfaces from predefined .pkl file"""
itf = utils.loadInterfaces(filename = opt.input)

"""Build interface using default filename"""
for t in opt.translate:
    if opt.translate == [-1]:
        T = None
        t = ""
    else:
        T = t

    itf.exportInterface(idx = opt.index, z_1 = opt.repeat_bottom, z_2 = opt.repeat_top,\
                        d = opt.distance, format = "lammps", translation = T,\
                        surface = opt.bottom_surface, anchor = "@%s" % t,\
                        verbose = opt.verbose, vacuum = opt.vacuum)
