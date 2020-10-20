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

    elif lattice == "WC0001_V":
        """HCP WC Lattice with 0001 surface relaxed using LAMMPS"""
        """Updated 2020-08-25, ENCUT=520, KPTS=21x21x21"""
        cell = np.array([[2.919028, -1.459509, 0       ],\
                         [       0,  2.527943, 0       ],\
                         [       0,         0, 2.845132]])

        pos = np.array([[       0,        0,        0],\
                        [1.459476, 0.842629, 1.422566]])

        spec = np.chararray(shape = 2, itemsize = 2)
        spec[0] = 'W'
        spec[1] = 'C'

        mass = np.array([183.84, 12.0107])


    elif lattice == "WC10-10_T_L":
        """HCP WC Lattice with 10-10 (T) surface relaxed using VASP"""
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


    elif lattice == "WC10-10_T_V":
        """HCP WC Lattice with 10-10 (T) surface relaxed using VASP"""
        """Updated 2020-08-25, ENCUT=520, KPTS=17x17x17"""
        cell = np.array([[2.918911, 0       , 0       ],\
                         [0       , 2.844869, 0       ],\
                         [0       , 0       , 5.056064]])

        pos = np.array([[1.45946, 0      , 1.68539],\
                        [      0, 0      , 4.21342],\
                        [1.45946, 1.42243, 0      ],\
                        [      0, 1.42243, 2.52803]])

        spec = np.chararray(shape = 4, itemsize = 2)
        spec[0] = 'C'
        spec[1] = 'C'
        spec[2] = 'W'
        spec[3] = 'W'

        mass = np.array([12.0107, 12.0107, 183.84, 183.84])


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
        
    elif lattice == "W111":
        print("Do W111")


    return cell, pos, spec, mass
