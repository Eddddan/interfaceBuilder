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

parser.add_argument("-i", "--input", type = str,\
                    help = "File containing all interfaces")

parser.add_argument("-idx", "--index", type = int,\
                    help = "Index of the interface to be built")

bs = ["0001", "10-10"]
parser.add_argument("-bs", "--bottom_surface", required = True, choices = bs,\
                    help = "Index of bottom surface")

ts = ["100", "110", "111"]
parser.add_argument("-ts", "--top_surface", required = False, choices = ts,\
                    help = "Index of top surface")

parser.add_argument("-t", "--translate", nargs = "+", default = [0], type = int,\
                    help = "Index of translation applied to the interface")

opt = parser.parse_args()

"""Load the interfaces from predefined .pkl file"""
itf = utils.loadInterfaces(filename = opt.input)

"""Build interface using default filename"""
for t in opt.translate:
    if opt.translate == [0]:
        t = None

    itf.exportInterface(idx = opt.index, z_1 = 6, z_2 = 6, d = 2, format = "lammps",\
                        translate = t, surface = opt.bottom_surface)
