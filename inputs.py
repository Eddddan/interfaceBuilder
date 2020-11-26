#!/usr/bin/env python3

import numpy as np

def getInputs(lattice):
    """Add predefined cell information to enable loading into structure objects

    add at the end;
    
    elif lattice == <speciy_name>:

        cell = np.ndarray[3, 3] #Unit cell

        pos = np.ndarray[[x, y, z]] #x, y, z positions each row one atom

        spec = np.chararray(shape = <nr_atoms>, itemsize = 2) #Chararray to hold elements
        spec[atom_type_1] = <atom_type_1> #Add element information to correct atom index
        spec[atom_type_2] = <atom_type_2>
        etc...

        mass = np.ndarray[<nr_atoms>] #Array of atom masses

    """

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


    elif lattice == "W100_V":
        """BCC W Lattice with 100 surface relaxed using VASP"""
        """Updated 2020-08-25, ENCUT=520, KPTS=21x21x21"""
        cell = np.array([[3.171856,        0, 0      ],\
                         [       0, 3.171856, 0      ],\
                         [       0,        0, 3.171856]])
        
        pos = np.array([[       0,        0,        0],\
                        [1.585928, 1.585928, 1.585928]])
        
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


    elif lattice == "W110_V":
        """BCC W Lattice with 110 surface relaxed using VASP"""
        """Updated 2020-08-25, ENCUT=520, KPTS=21x21x21"""
        cell = np.array([[3.172109, 1.586055, 0       ],\
                         [       0, 2.242216, 0       ],\
                         [       0,        0, 4.487154]])
        
        pos = np.array([[       0, 0,  0       ],\
                        [1.586055, 0,  2.243577]])
        
        spec = np.chararray(shape = 2, itemsize = 2)
        spec[:] = 'W'

        mass = np.array([183.84, 183.84])


    return cell, pos, spec, mass
