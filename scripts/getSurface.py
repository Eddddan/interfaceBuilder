#!/usr/bin/env python3

import argparse
import numpy as np

from interfaceBuilder import interface
from interfaceBuilder import structure
from interfaceBuilder import file_io
from interfaceBuilder import inputs
from interfaceBuilder import utils

"""Parse inputs"""
parser = argparse.ArgumentParser(description = "Script for retreving surfaces",\
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                 prog = "getSurface")

parser.add_argument("--input", type = str, required = True, metavar = "I",\
                    help = "File containing all interfaces")

parser.add_argument("--index", type = int, required = True, metavar = "IDX",\
                    help = "Index of the interface to be built")

surf = [1, 2]
parser.add_argument("--surface", required = True, choices = surf, default = 2, metavar = "S",\
                    type = int, help = "Build surface 1 (bottom) or surace 2 (top)")

parser.add_argument("--strain", action = "store_true",\
                    help = "Strain the surface (if surace 2) to match surface 1")

parser.add_argument("--vacuum", type = float, default = 10.0, metavar = "VAC",\
                    help = "Vacuum added above the surface")

parser.add_argument("--output", type = str, default = None, metavar = "O",\
                    help = "Name of the output file")

parser.add_argument("-v", "--verbose", action = "count", default = 0,\
                    help = "Print extra information")

parser.add_argument("--repeat", type = int, default = 3, metavar = "R",\
                    help = "Repetitions in the Z direction")

parser.add_argument("--alt_base", action = "store_true",\
                    help = "Use predefined alternative base")

fmt = ["lammps", "vasp"]
parser.add_argument("--format", choices = fmt, default = "lammps", metavar = "F", type = str.lower,\
                    help = "Format in which to export the interface")

parser.add_argument("--kpts_version", type = str, default = None, metavar = "KV",\
                    help = "Option for writing a kpoints file")

parser.add_argument("-k_n1", "--kpts_n1", default = None, type = int, metavar = "KN1",\
                    help = "Option for the nr of kpoints in direction N1")

parser.add_argument("--kpts_n2", default = None, type = int, metavar = "KN2",\
                    help = "Option for the nr of kpoints in direction N2")

parser.add_argument("--kpts_n3", default = None, type = int, metavar = "KN3",\
                    help = "Option for the nr of kpoints in direction N3")

parser.add_argument("-k_d", "--kpts_density", default = 1, type = int, metavar = "KD",\
                    help = "Option for the density of kpoints in each reciprical direction")

opt = parser.parse_args()

if opt.kpts_version is None:
    kpts = None
else:
    kpts = {"version": opt.kpts_version, "N1": opt.kpts_n1, "N2": opt.kpts_n2,\
            "N3": opt.kpts_n3, "density": opt.kpts_density}

"""Load the interfaces from predefined .pkl file"""
itf = utils.loadInterfaces(filename = opt.input)

"""Build interface using default filename"""
itf.exportSurface(idx = opt.index, z = opt.repeat, format = opt.format, filename = opt.output,\
                  surface = opt.surface, verbose = opt.verbose, ab = opt.alt_base,\
                  vacuum = opt.vacuum, strained = opt.strain, kpoints = kpts)
