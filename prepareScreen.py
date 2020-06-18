#!/usr/bin/env python3

import utils
import inputs
import file_io
import argparse
import interface
import structure
import numpy as np

"""Create structures by loading from the predefined input file"""
a = structure.Structure(load_from_input = "WC10-10_T_L")
b = structure.Structure(load_from_input = "W100_L")

"""Build an interface from the two structures"""
q = interface.Interface(structure_a = a, structure_b = b)

"""Search the geometries to find interface matches, given the supplied criteria"""
n = 12; m = 12
q.matchCells(dTheta = 2, n_max = n, m_max = m, max_strain = 1, min_angle = 10,\
             limit_asr = 75000, remove_asd = True, max_atoms = 5000)

"""Change elements to enable 3 diferent types, W1 (WC), C (WC), W2 (bcc W)"""
q.changeBaseElements(change = {"from": "W", "to": "W1", "mass": 183.84}, cell = 1)
q.changeBaseElements(change = {"from": "W", "to": "W2", "mass": 183.84}, cell = 2)

"""Hexplot all combinations and save fig"""
q.hexPlotCombinations(save = "SB_hexbin_asr.pdf")

"""Find 5000 matches given a fit following (min_strain) to (min_strain,min_atoms)"""
C, E = q.getAtomStrainMatches(matches = 5000)
ratio = q.getAtomStrainRatio(const = C, exp = E)
q.indexSortInterfaces(np.argsort(ratio))

"""Plot the matches"""
q.plotCombinations(const = C, exp = E, save = "SB_combinations.png", format = "png", dpi = 500)

"""Save interfaces to .pkl file"""
fname = "Interfaces_1010_T_C_100_%ix%i_lammps.pkl" % (n, m)
q.saveInterfaces(filename = fname)

"""Export a single interface to test"""
q.exportInterface(idx = 2, d = 2.1, z_1 = 3, z_2 = 3, translation = 2, surface = "10-10",\
                  vacuum = 10, verbose = 1)

"""Swap elements in cell_1 to change the termination"""
q.changeBaseElements(swap = {"swap_1": "W1", "swap_2": "C"}, cell = 1)

"""Save interfaces to .pkl file"""
fname = "Interfaces_1010_T_W_100_%ix%i_lammps.pkl" % (n, m)
q.saveInterfaces(filename = fname)

"""Export a single interface to test"""
q.exportInterface(idx = 2, d = 2.3, z_1 = 3, z_2 = 3, translation = 3, surface = "10-10",\
                  vacuum = 10, verbose = 1)
