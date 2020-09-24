#!/usr/bin/env python3

import utils
import inputs
import file_io
import argparse
import interface
import structure
import numpy as np


"""This script prepares 8 different types of interfaces (0001 or 10-10(T)) 
   against (100 or 110) with both C and W terminations of the bottom WC cell.

   All structure parameters are in this case stored in the input.py file.
   
   It searches 12 by 12 repititions of the top cell and matches it to the
   best possible match of the bottom cell for each rotation between the 2 cells
   at intervalls of 2 degrees. We limit the nr of outputs to 5000 (millions of 
   interfaces are initially found) interfaces with the main critera or selection
   beeing a ratio between strain and the number of atoms.

   The interface objects are saved to .pkl files which can be easily loaded by
   typing i = utils.loadInterfaces("Name-of-the-saved-file").

   Simply run this script as ./prepareScreen in the same directory as the 
   other files
"""   

A = ["WC0001_L", "WC10-10_T_L"]
B = ["W100_L", "W110_L"]

name_A = ["0001", "1010_T"]
name_B = ["100", "110"]

l = 1
n = 12; m = 12
total_nr = 5000

for i, i_item in enumerate(A):
    for ii, ii_item in enumerate(B):

        """Create structures by loading from the predefined input file"""
        a = structure.Structure(load_from_input = i_item)
        b = structure.Structure(load_from_input = ii_item)

        """Build an interface from the two structures"""
        q = interface.Interface(structure_a = a, structure_b = b)

        """Search the geometries to find interface matches, given the supplied criteria"""
        q.matchCells(dTheta = 2, n_max = n, m_max = m, max_strain = 1, min_angle = 10,\
                         limit_asr = 75000, remove_asd = True, max_atoms = 5000)

        """Change elements to enable 3 diferent types, W1 (WC), C (WC), W2 (bcc W)"""
        q.changeBaseElements(change = {"from": "W", "to": "W1", "mass": 183.84}, cell = 1)
        q.changeBaseElements(change = {"from": "W", "to": "W2", "mass": 183.84}, cell = 2)

        """Hexplot all combinations and save fig"""
        q.hexPlotCombinations(save = "SB_hexbin_asr.pdf")

        """Find 5000 matches given a fit following (min_strain) to (min_strain,min_atoms)"""
        C, E = q.getAtomStrainMatches(matches = total_nr)
        ratio = q.getAtomStrainRatio(const = C, exp = E)
        q.indexSortInterfaces(np.argsort(ratio))

        """Keep only the first 5000 interfaces"""
        keep = np.zeros(q.atoms.shape[0], dtype = bool)
        keep[:total_nr] = True
        q.deleteInterfaces(keep)

        """Plot the matches"""
        q.plotCombinations(const = C, exp = E, save = "SB_combinations.png", format = "png", dpi = 500)

        """Save interfaces to .pkl file"""
        fname = "%i_Interfaces_%s_%s_%s_%ix%i_lammps.pkl" % (l, name_A[i], "C", name_B[ii], n, m)
        q.saveInterfaces(filename = fname)
        l += 1

        """Export a single interface to test"""
        q.exportInterface(idx = 2, d = 2.1, z_1 = 3, z_2 = 3, translation = 2, surface = "10-10",\
                              vacuum = 10, verbose = 1)

        """Swap elements in cell_1 to change the termination"""
        q.changeBaseElements(swap = {"swap_1": "W1", "swap_2": "C"}, cell = 1)

        """Save interfaces to .pkl file"""
        fname = "%i_Interfaces_%s_%s_%s_%ix%i_lammps.pkl" % (l, name_A[i], "W", name_B[ii], n, m)
        q.saveInterfaces(filename = fname)
        l += 1

        """Export a single interface to test"""
        q.exportInterface(idx = 2, d = 2.3, z_1 = 3, z_2 = 3, translation = 3, surface = "10-10",\
                              vacuum = 10, verbose = 1)
