#!/usr/bin/env python3

import sys
import inputs
import file_io
import argparse
import interface
import structure
import utils as ut
import numpy as np


"""Parse inputs"""
parser = argparse.ArgumentParser(description = "Script for getting nearest neighbors from specified "\
                                 "atoms in a Structure",\
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                 prog = "getNN")

parser.add_argument("-i", "--input", type = str, required = True,\
                    help = "File containing the structure")

parser.add_argument("-ii", "--i_int", type = int, required = True,\
                    help = "Interface index")

parser.add_argument("-it", "--i_trans", type = str, required = True,\
                    help = "Translation index")

formats = ["vasp", "lammps", "xyz", "eon"]
parser.add_argument("-f", "--format", type = str, required = True,\
                    choices = formats, help = "Input file format")

parser.add_argument("-n", "--neighbors", type = int, default = 8,\
                    help = "Nr of NN to find")

parser.add_argument("-s", "--save", type = str, default = "NN_log.txt",\
                    help = "File to save the data to")

parser.add_argument("-v", "--verbose", action = "count", default = 0,\
                    help = "Print extra information")

opt = parser.parse_args()

"""Try opening the file. If it doesnt exist the sim program exited with an error"""
try:
    s = structure.Structure(load_from_file = opt.input, format = opt.format)
except FileNotFoundError:
    """Except FileNotFoundError and print all 0 to the NN file (along with index...)"""
    string  = "EXCEPTION: File %s not found" % opt.input
    ut.infoPrint(string)

    with open(opt.save, 'a') as f:
        string_head = "%s %s %s " % (opt.i_int, opt.i_trans, "0")
        string_mean = " ".join([str(i) for i in np.zeros(opt.neighbors)]) + " "
        string_std = " ".join([str(i) for i in np.zeros(opt.neighbors)])
    
        f.write(string_head)
        f.write(string_mean)
        f.write(string_std)
        f.write("\n")
    
    """Exit the script"""
    sys.exit()

idx, elem, type_i = s.getElementIdx()
s.dir2car()
max_0 = np.max(s.pos[idx[0]][:, 2])
max_1 = np.max(s.pos[idx[1]][:, 2])

"""Calculate from the highest element in the bottom surface to other species"""
if max_0 > max_1:
    i_from = idx[0][s.pos[idx[0], 2] > np.max(s.pos[idx[0], 2]) - 1]
    i = 0
    max = max_0
else:
    i_from = idx[1][s.pos[idx[1], 2] > np.max(s.pos[idx[1], 2]) - 1]
    i = 1
    max = max_1



if opt.verbose > 0:
    string = "Calculating NN for file: %s, with elements: %s(%i), %s(%i), %s(%i)"\
              % (opt.input, elem[0], type_i[0], elem[1], type_i[1], elem[2], type_i[2])
    ut.infoPrint(string)

    string = "Calculating from element: %s(%i) at z>: %.2f, nr of atoms: %i"\
             % (elem[i], type_i[i], max - 1, np.shape(i_from)[0])
    ut.infoPrint(string)

with open(opt.save, 'a') as f:

    dist_mean, dist_std = s.getNearestNeighborCollection(idx = i_from, NN = opt.neighbors,\
                                                         verbose = opt.verbose)

    string_head = "%s %s %s " % (opt.i_int, opt.i_trans, type_i[i])
    string_mean = " ".join([str(i) for i in np.round(dist_mean, 3)]) + " "
    string_std = " ".join([str(i) for i in np.round(dist_std, 3)])
    

    f.write(string_head)
    f.write(string_mean)
    f.write(string_std)
    f.write("\n")
