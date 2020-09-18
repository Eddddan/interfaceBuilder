#!/usr/bin/env python3

import utils
import inputs
import file_io
import argparse
import interface
import structure
import numpy as np


"""Parse inputs"""
parser = argparse.ArgumentParser(description = "Script for setting the work of adhesion "\
                                 "from an array of data",\
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                 prog = "set_wsep_array")

parser.add_argument("-i", "--input", type = str, required = True,\
                    help = "File containing all interfaces")

parser.add_argument("-v", "--verbose", action = "count", default = 0,\
                    help = "Print extra information")

parser.add_argument("-ai", "--array_input", type = str, nargs = "+", required = True,\
                    help = "File containing energy/area data shape(N+1,4), (idx, E_t0, E_t1,...) "\
                    "with a header containing the translations")

fmt = ["lammps", "vasp"]
parser.add_argument("-f", "--format", choices = fmt, default = "lammps",\
                    help = "Set lammps data or vasp data")

opt = parser.parse_args()

"""Load file containing the interfaces"""
itf = utils.loadInterfaces(filename = opt.input, verbose = opt.verbose)

for i in opt.array_input:
    """Load data from array_inputs, should be formatted as shape(N,3) with cols (idx, translation, energy)"""
    with open(i, 'r') as ai:
        trans = [np.int(i) for i in ai.readline().split()]
        data = np.loadtxt(ai)

    if opt.verbose > 1:
        string = "Reading file: %s" % i
        ut.infoPrint(string)

    """Cast index as int"""
    index = data[:, 0].astype(np.int)

    """Set the energy"""
    itf.setWsepArray(idx = index, translation = trans, w_sep = data[:, 1:], version = opt.format)

"""Save the interfaces"""
itf.saveInterfaces(filename = opt.input, verbose = opt.verbose)
