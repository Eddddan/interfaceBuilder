#!/usr/bin/env python3

import sys
import argparse
import numpy as np

from interfaceBuilder import interface
from interfaceBuilder import utils

"""Parse inputs"""
parser = argparse.ArgumentParser(description = "Script for setting array properties, i.e. " +\
                                               "w_sep, w_sep_vasp, w_sep_strain, w_sep_strain_vasp, "+\
                                               "e_int, e_int_vasp",\
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                 prog = "setProperty")

parser.add_argument("-i", "--input", type = str, required = True, metavar = "I",\
                    help = ".pkl file containing the saved interface object")

parser.add_argument("-idx", "--index", type = int, metavar = "IDX",\
                    help = "Index of the interface", default = None)

parser.add_argument("-val", "--value", type = float, metavar = "VAL",\
                    help = "Value to set", default = None)

parser.add_argument("-p", "--property", type = str, required = True,\
                    metavar = "P", help = "Which property to set")

parser.add_argument("-t", "--translation", type = int, metavar = "T",\
                    help = "Index of the supplied translation", default = None)

parser.add_argument("-f", "--file", type = str, default = None, nargs = "+", metavar = "F",\
                    help = "File containing a header specifying translations and a " +\
                    "first column containing index and corresponding values")

parser.add_argument("-v", "--verbose", action = "count", default = 0,\
                    help = "Print extra information")

opt = parser.parse_args()

"""Load the interfaces from predefined .pkl file"""
itf = utils.loadInterfaces(filename = opt.input)

"""Use the file if supplied, otherwise set the single value"""
if opt.file is not None:
    for i in opt.file:
        with open(i, 'r') as f:
            t = [np.int(j) for j in f.readline().split()]
            data = np.loadtxt(f)

        idx = data[:, 0].astype(np.int)
        val = data[:, 1:]
        itf.setArrayProperty(prop = opt.property, idx = idx,\
                             t = t, value = val, verbose = opt.verbose)
else:
    if opt.translation is not None and opt.index is not None and opt.value is not None:
        itf.setArrayProperty(prop = opt.property, idx = opt.index, t = opt.translation,\
                                 value = opt.value, verbose = opt.verbose)
    else:
        string = "Translation, index and value needs to be specified"
        print("=" * len(string) + "\n" + string + "\n" + "=" * len(string))
        sys.exit(0)

"""Save the updated interface data set"""
itf.saveInterfaces(filename = opt.input, verbose = opt.verbose)
