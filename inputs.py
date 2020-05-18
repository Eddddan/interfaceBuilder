#!/usr/bin/env python3

import numpy as np

def getInputs(lattice):

    if lattice == "WC0001_L":
        """HCP WC Lattice with 0001 surface relaxed using LAMMPS"""
        """Updated 2020-05-13"""
        cell = np.array([[2.91720,  -1.4586, 0       ],\
                             [      0, 2.512963, 0       ],\
                             [      0,        0, 2.848620]])

        pos = np.array([[       0,        0,        0],\
                            [1.450715, 0.837571, 1.424310]])

        spec = np.chararray(shape = 2, itemsize = 2)
        spec[0] = 'W'
        spec[1] = 'C'

        mass = np.array([183.84, 12.0107])



    elif lattice == "WC10-10":
        print("Do WC10-10")



    elif lattice == "W100_L":
        """BCC W Lattice with 100 surface relaxed using LAMMPS"""
        """Updated 2020-05-13"""
        cell = np.array([[3.164917,        0, 0       ],\
                             [       0, 3.164917, 0       ],\
                             [       0,        0, 3.164917]])
        
        pos = np.array([[       0,        0,        0],\
                            [1.582459, 1.582459, 1.582459]])
        
        spec = np.chararray(shape = 2, itemsize = 2)
        spec[:] = 'W'

        mass = np.array([183.84, 183.84])


        
    elif lattice == "W110":
        print("Do W110")


        
    elif lattice == "W111":
        print("Do W111")



    return cell, pos, spec, mass
