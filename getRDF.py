#!/usr/bin/env python3

import utils as ut
import inputs
import file_io
import argparse
import interface
import structure
import numpy as np


"""Parse inputs"""
parser = argparse.ArgumentParser(description = "Script for getting RDF from a Structure",\
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter,\
                                 prog = "getRDF")

parser.add_argument("-i", "--input", type = str, required = True,\
                    help = "File containing the structure")

parser.add_argument("-ii", "--i_int", type = int, required = True,\
                    help = "Interface index")

parser.add_argument("-it", "--i_trans", type = str, required = True,\
                    help = "Translation index")

formats = ["vasp", "lammps", "xyz", "eon"]
parser.add_argument("-f", "--format", type = str, required = True,\
                    choices = formats, help = "Input file format")

parser.add_argument("-r", "--radius", type = float, default = 5.5,\
                    help = "Radial cut-off")

parser.add_argument("-dr", "--radial_step", type = float, default = 0.1,\
                    help = "Radial step size")

parser.add_argument("-s", "--save", type = str, default = "RDF_log.txt",\
                    help = "File to save the data to")

parser.add_argument("-v", "--verbose", action = "count", default = 0,\
                    help = "Print extra information")

opt = parser.parse_args()

s = structure.Structure(load_from_file = opt.input, format = opt.format)

idx, elem, type_i = s.getElementIdx()
s.dir2car()
max_0 = np.max(s.pos[idx[0]][:, 2])
max_1 = np.max(s.pos[idx[1]][:, 2])
    
"""Calculate from the highest element in the bottom surface to otehr species"""
if max_0 > max_1:
    i_from = idx[0][s.pos[idx[0], 2] > np.max(s.pos[idx[0], 2]) - 1]
    i_to = 1
    i_from_idx = 0
else:
    i_from = idx[1][s.pos[idx[1], 2] > np.max(s.pos[idx[1], 2]) - 1]
    i_to = 0
    i_from_idx = 1

if opt.verbose > 0:
    string = "Calculating RDF for file: %s" % (opt.input)
    ut.infoPrint(string)

start = 1.4
prec = 3
with open(opt.save, 'a') as f:

    cnt, bin, tot = s.getRDF(idx = i_from, idx_to = idx[i_from_idx], r = opt.radius, dr = opt.radial_step,\
                             verbose = 1)

    cnt = np.round(cnt[bin > start], prec)
    tot = np.round(tot[bin > start], prec)
    bin = np.round(bin[bin > start], prec)

    string_head = "%s %s %s " % (opt.i_int, opt.i_trans, type_i[i_from_idx])
    string_bin = " ".join([str(i) for i in bin]) + " "
    string_cnt = " ".join([str(i) for i in cnt]) + " "
    string_tot = " ".join([str(i) for i in tot])

    f.write(string_head)
    f.write(string_bin)
    f.write(string_cnt)
    f.write(string_tot)
    f.write("\n")

    cnt, bin, tot = s.getRDF(idx = i_from, idx_to = idx[i_to], r = opt.radius, dr = opt.radial_step,\
                             verbose = 1)

    cnt = np.round(cnt[bin > start], prec)
    tot = np.round(tot[bin > start], prec)
    bin = np.round(bin[bin > start], prec)

    string_head = "%s %s %s " % (opt.i_int, opt.i_trans, type_i[i_to])
    string_bin = " ".join([str(i) for i in bin]) + " "
    string_cnt = " ".join([str(i) for i in cnt]) + " "
    string_tot = " ".join([str(i) for i in tot])

    f.write(string_head)
    f.write(string_bin)
    f.write(string_cnt)
    f.write(string_tot)
    f.write("\n")

    cnt, bin, tot = s.getRDF(idx = i_from, idx_to = idx[2], r = opt.radius, dr = opt.radial_step,\
                             verbose = 1)

    cnt = np.round(cnt[bin > start], prec)
    tot = np.round(tot[bin > start], prec)
    bin = np.round(bin[bin > start], prec)

    string_head = "%s %s %s " % (opt.i_int, opt.i_trans, type_i[2])
    string_bin = " ".join([str(i) for i in bin]) + " "
    string_cnt = " ".join([str(i) for i in cnt]) + " "
    string_tot = " ".join([str(i) for i in tot])

    f.write(string_head)
    f.write(string_bin)
    f.write(string_cnt)
    f.write(string_tot)
    f.write("\n")
