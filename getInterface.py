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

fmt = ["lammps", "vasp"]
parser.add_argument("-f", "--format", choices = fmt, default = "lammps",\
                    help = "Format in which to export the interface")

parser.add_argument("-t", "--translate", nargs = "+", default = [-1], type = int,\
                    help = "Index of translation applied to the interface, 0-based")

parser.add_argument("-d", "--distance", type = float, default = 2.0,\
                    help = "Distance between the surfaces")

parser.add_argument("-vac", "--vacuum", type = float, default = 10.0,\
                    help = "Vacuum added above the interface")

parser.add_argument("-v", "--verbose", action = "count", default = 0,\
                    help = "Print extra information")

parser.add_argument("-o", "--output", type = str, default = None,\
                    help = "Name of the output file")

parser.add_argument("-zb", "--repeat_bottom", type = int, default = 3,\
                    help = "Repetitions in the Z direction for the bottom surface")

parser.add_argument("-zt", "--repeat_top", type = int, default = 3,\
                    help = "Repetitions in the Z direction for the top surface")

parser.add_argument("-ab", "--alt_base", nargs = "+", default = None, type = float,\
                    help = "Alternate base to use in construction, [11, 21, 12, 22]")

parser.add_argument("-k_v", "--kpts_version", type = str, default = None,\
                    help = "Option for writing a kpoints file")

parser.add_argument("-k_n1", "--kpts_n1", default = None, type = int,\
                    help = "Option for the nr of kpoints in direction N1")

parser.add_argument("-k_n2", "--kpts_n2", default = None, type = int,\
                    help = "Option for the nr of kpoints in direction N2")

parser.add_argument("-k_n3", "--kpts_n3", default = None, type = int,\
                    help = "Option for the nr of kpoints in direction N3")

parser.add_argument("-k_d", "--kpts_density", default = 1, type = int,\
                    help = "Option for the density of kpoints in each reciprical direction")

opt = parser.parse_args()

"""Load the interfaces from predefined .pkl file"""
itf = utils.loadInterfaces(filename = opt.input)

if opt.kpts_version is None:
    kpts = None
else:
    kpts = {"version": opt.kpts_version, "N1": opt.kpts_n1, "N2": opt.kpts_n2,\
            "N3": opt.kpts_n3, "density": opt.kpts_density}

if opt.alt_base is not None:
    alt_base = np.array(opt.alt_base)
    opt.alt_base = alt_base.reshape((2, 2)).T

"""Build interface using default filename"""
for t in opt.translate:
    if opt.translate == [-1]:
        T = None
        t = ""
    else:
        T = t

    itf.exportInterface(idx = opt.index, z_1 = opt.repeat_bottom, z_2 = opt.repeat_top,\
                        d = opt.distance, format = opt.format, translation = T,\
                        surface = opt.bottom_surface, anchor = "@%s" % t,\
                        verbose = opt.verbose, vacuum = opt.vacuum,\
                        alt_base = opt.alt_base, filename = opt.output, kpoints = kpts)
