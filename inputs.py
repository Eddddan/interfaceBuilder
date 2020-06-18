#!/usr/bin/env python3

import numpy as np

def getInputs(lattice):

    if lattice == "WC0001_L":
        """HCP WC Lattice with 0001 surface relaxed using LAMMPS"""
        """Updated 2020-05-13"""
        cell = np.array([[2.91658918, -1.45829459, 0         ],\
                         [       0,    2.52584033, 0         ],\
                         [       0,             0, 2.81210207]])

        pos = np.array([[       0,        0,        0],\
                        [1.45822172, 0.84190490, 1.40605103]])

        spec = np.chararray(shape = 2, itemsize = 2)
        spec[0] = 'W'
        spec[1] = 'C'

        mass = np.array([183.84, 12.0107])


    elif lattice == "WC10-10_T_L":
        """HCP WC Lattice with 10-10 (T) surface relaxed using LAMMPS"""
        """Updated 2020-06-18"""
        cell = np.array([[2.91658918,           0, 0         ],\
                         [         0,  2.81210207, 0         ],\
                         [         0,           0, 5.05168065]])

        pos = np.array([[1.45829459,          0, 1.68389355],\
                        [1.45829459, 1.40605103,          0],\
                        [         0,          0, 4.20973388],\
                        [         0, 1.40605103, 2.52584033]])

        spec = np.chararray(shape = 4, itemsize = 2)
        spec[0] = 'C'
        spec[1] = 'W'
        spec[2] = 'C'
        spec[3] = 'W'

        mass = np.array([12.0107, 183.84, 12.0107, 183.84])


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

        
    elif lattice == "W110_L":
        """BCC W Lattice with 110 surface relaxed using LAMMPS"""
        """Updated 2020-06-18"""
        cell = np.array([[3.164917, 1.582459, 0       ],\
                         [       0, 2.237935, 0       ],\
                         [       0,        0, 4.475869]])
        
        pos = np.array([[0       , 0,  0       ],\
                        [1.582459, 0,  2.237935]])
        
        spec = np.chararray(shape = 2, itemsize = 2)
        spec[:] = 'W'

        mass = np.array([183.84, 183.84])
        
    elif lattice == "W111":
        print("Do W111")


    return cell, pos, spec, mass
