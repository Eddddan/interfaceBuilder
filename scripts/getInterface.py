#!/usr/bin/env python3

import argparse
import numpy as np

from interfaceBuilder import interface
from interfaceBuilder import structure
from interfaceBuilder import file_io
from interfaceBuilder import inputs
from interfaceBuilder import utils


def choice_format(choice):
    return choice.lower()

"""Parse inputs"""
parser = argparse.ArgumentParser(description = "Script for retreving interfaces",\
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                 prog = "getInterface")

parser.add_argument("--input", type = str, required = True, metavar = "I",\
                    help = "File containing all interfaces")

parser.add_argument("--index", type = int, required = True, metavar = "IDX",\
                    help = "Index of the interface to be built")

bs = ["0001", "10-10", "b100", "b110", "f100", "f110"]
parser.add_argument("--bottom", required = True, type = str.lower, choices = bs, metavar = "B",\
                    help = "Index of bottom surface")

fmt = ["lammps", "vasp"]
parser.add_argument("--format", choices = fmt, default = "lammps", metavar = "F", type = str.lower,\
                    help = "Format in which to export the interface")

parser.add_argument("--translate", nargs = "+", default = [-1], type = int, metavar = "T",\
                    help = "Index of translation applied to the interface, 0-based")

parser.add_argument("--distance", type = float, default = 2.0, metavar = "D",\
                    help = "Distance between the surfaces")

parser.add_argument("--vacuum", type = float, default = 10.0, metavar = "VAC",\
                    help = "Vacuum added above the interface")

parser.add_argument("-v", "--verbose", action = "count", default = 0,\
                    help = "Print extra information")

parser.add_argument("--output", type = str, default = None, metavar = "O",\
                    help = "Name of the output file")

parser.add_argument("--n_bottom", type = int, default = 3, metavar = "NB",\
                    help = "Repetitions in the Z direction for the bottom surface")

parser.add_argument("--n_top", type = int, default = 3, metavar = "NT",\
                    help = "Repetitions in the Z direction for the top surface")

parser.add_argument("--alt_base", action = "store_true",\
                    help = "Use predefined alternative base")

parser.add_argument("--kpts_version", type = str, default = None, metavar = "KV", \
                    help = "Option for writing a kpoints file")

parser.add_argument("--kpts_n1", default = None, type = int, metavar = "KN1",\
                    help = "Option for the nr of kpoints in direction N1")

parser.add_argument("--kpts_n2", default = None, type = int, metavar = "KN2",\
                    help = "Option for the nr of kpoints in direction N2")

parser.add_argument("--kpts_n3", default = None, type = int, metavar = "KN3",\
                    help = "Option for the nr of kpoints in direction N3")

parser.add_argument("--kpts_density", default = 1, type = int, metavar = "KD",\
                    help = "Option for the density of kpoints in each reciprical direction")

opt = parser.parse_args()

"""Load the interfaces from predefined .pkl file"""
itf = utils.loadInterfaces(filename = opt.input)

if opt.kpts_version is None:
    kpts = None
else:
    kpts = {"version": opt.kpts_version, "N1": opt.kpts_n1, "N2": opt.kpts_n2,\
            "N3": opt.kpts_n3, "density": opt.kpts_density}

"""Build interface using default filename"""
for t in opt.translate:
    if opt.translate == [-1]:
        T = None
        t = ""
    else:
        T = t

    itf.exportInterface(idx = opt.index, z_1 = opt.n_bottom, z_2 = opt.n_top,\
                        d = opt.distance, format = opt.format, translation = T,\
                        surface = opt.bottom, anchor = "@%s" % t,\
                        verbose = opt.verbose, vacuum = opt.vacuum,\
                        filename = opt.output, kpoints = kpts, ab = opt.alt_base)
