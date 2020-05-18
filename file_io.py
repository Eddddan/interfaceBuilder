#!/usr/bin/env python3

import sys
import numpy as np



def len2mat(vec, ang, prec = 10):
    """
    Transforms cell lengths and angles to cell vectors.

    vec in order [a, b, c]

    ang in order [alpha, beta, gamma], (conventionally defined)

    prec = round to this precision. Sin and Cos are numeric i.e.
           cos(90) is ~X*10^-17 (something like that...)
    """

    """Fix precision as sin and cos are numeric"""
    prec = 7

    """M = [A, B, C]"""
    mat = np.zeros((3, 3))
    """A = [ax; 0; 0]"""
    mat[0, 0] = vec[0]
    """B = [bx; by; 0]"""
    mat[0, 1] = vec[1] * np.round(np.cos(np.deg2rad(ang[2])), prec)
    mat[1, 1] = vec[1] * np.round(np.sin(np.deg2rad(ang[2])), prec)
    """C = [cx; cy; cz]"""
    mat[0, 2] = vec[2] * np.round(np.cos(np.deg2rad(ang[1])), prec)
    mat[1, 2] = vec[2] * np.round((np.cos(np.deg2rad(ang[0]))  - \
                                   np.cos(np.deg2rad(ang[2]))  * \
                                   np.cos(np.deg2rad(ang[1]))) / \
                                   np.sin(np.deg2rad(ang[2])), prec)
    mat[2, 2] = np.round(np.sqrt(vec[2]**2 - mat[0, 2]**2 - mat[1, 2]**2), prec)

    return mat
    


def mat2LammpsBox(mat, prec = 10):
    """Function for transforming a set of basis vectors to
       a lammps simulation box"""

    
    lx = mat[0, 0]
    ly = mat[1, 1]
    lz = mat[2, 2]

    xy = mat[0, 1]
    xz = mat[0, 2]
    yz = mat[1, 2]
    
    x_lo_b = np.min([0, lx]) + np.min([0, xy, xz, xy + xz])
    x_hi_b = np.max([0, lx]) + np.max([0, xy, xz, xy + xz])

    y_lo_b = np.min([0, ly]) + np.min([0, yz])
    y_hi_b = np.max([0, ly]) + np.max([0, yz]) 

    z_lo_b = np.min([0, lz])
    z_hi_b = np.max([0, lz])

    box = [x_lo_b, x_hi_b, y_lo_b, y_hi_b, z_lo_b, z_hi_b]

    return box


def readEON(filename):
    """Load EON geometry file"""

    """Lines to skip"""
    skip = [0, 1, 4, 5, 10]

    """Open and read the file"""
    with open(filename, 'r') as f:
        for i, line in enumerate(f):

            if i in skip:
                continue 
               
            if i == 2:
                """Read cell lengths"""
                vec = np.array([np.float(x) for x in line.split()])
            elif i == 3:
                """Angles in order alpha (yz), beta (xz), gamma (xy)"""
                ang = np.array([np.float(x) for x in line.split()[::-1]])
            elif i == 6:
                """Nr of diffefrent species"""
                M = np.int(line)
            elif i == 7:
                """Number of each species"""
                N = np.array([np.int(x) for x in line.split()], dtype = np.int)
                Nt = N.cumsum()
                n = 0; nt = 0
                pos = np.zeros((Nt[-1], 3))
                idx = np.zeros(Nt[-1])
            elif i == 8:
                """Mass"""
                mass = np.array([np.float(x) for x in line.split()])
            elif i == 9:
                """Atom type of species 1"""
                t = np.chararray(Nt[-1], itemsize = 2)
                t[:Nt[nt]] = line

            elif i > 10:
                l = line.split()
                if len(l) == 1:
                    t[Nt[nt] : Nt[nt + 1]] = line
                    nt += 1
                elif len(l) == 4:
                    continue
                else:
                    pos[n, :] = [np.float(x) for x in l[:3]]
                    idx[n] = np.float(l[4])
                    n += 1

    """Convert lengths and angles to cell vectors"""
    mat = len2mat(vec, ang)

    return mat, pos, t, idx

            

def readLAMMPS(filename):
    """Load LAMMPS geometry file"""
    print("loadLAMMPS")



def writeLAMMPS(filename, atoms):
    """Write a LAMMPS data file"""
    print("Write LAMMPS-file: %s" % filename)
    
    with open(filename, "w") as f:
        
        """Newline character"""
        nl = "\n"

        box = [0, atoms.cell[0, 0], 0, atoms.cell[1, 1], 0, atoms.cell[2, 2]]
        nr_atoms = atoms.pos.shape[0]
        nr_types = np.unique(atoms.type_i).shape[0]
        masses = np.unique(atoms.mass)

        """First a comment line"""
        f.write("LAMMPS-data file written from file_io.py\n")
        f.write(nl)

        f.write("%i atoms\n" % nr_atoms)
        f.write("%i atom types\n" % nr_types)
        f.write(nl)

        f.write("%13.7f %13.7f   xlo xhi\n" % (box[0], box[1]))
        f.write("%13.7f %13.7f   ylo yhi\n" % (box[2], box[3]))
        f.write("%13.7f %13.7f   zlo zhi\n" % (box[4], box[5]))
        f.write("%13.7f %13.7f %13.7f   xy xz yz\n"\
                % (atoms.cell[0, 1], atoms.cell[0, 2], atoms.cell[1, 2]))
        f.write(nl)

        f.write("Masses\n")
        f.write(nl)
        for i, item in enumerate(masses):
            f.write("%i %11.6f\n" % (i, item))
        f.write(nl)

        f.write("Atoms\n")
        f.write(nl)
        for i in range(nr_atoms):
            f.write("%5i %3i %13.7f %13.7f %13.7f\n"\
                    % (atoms.idx[i], atoms.type_i[i], atoms.pos[i, 0], atoms.pos[i, 1], atoms.pos[i, 2]))

    print("Done\n")


def readVASP(filename):
    """Load VASP geometry file"""
    print("loadVASP")



def readXYZ(filename):
    """Load XYZ geometry file"""
    print("loadXYZ")


def writeXYZ(filename, atoms):
    """Write a XYZ data file"""

    with open(filename, "w") as f:
        
        """Number of atoms"""
        f.write("%i\n" % atoms.pos.shape[0])

        """Info on the lattice and the mapping of the data in the file.
           In the format the lattice is supplied as ax,ay,az,bx,by,bz,cx,cy,cz"""
        c = atoms.cell 
        string = "Lattice=\"%f %f %f %f %f %f %f %f %f\" "\
                 "Properties=species:S:1:pos:R:3:masses:R:1 "\
                 "comment=Generated from file_io\" pbc=\"F F F\"\n" %\
                 (c[0, 0], c[1, 0], c[2, 0], c[0, 1], c[1, 1], c[2, 1],\
                  c[0, 2], c[1, 2], c[2, 2])
        f.write(string)

        for i in range(atoms.pos.shape[0]):
            string = "%5s %15.8f %15.8f %15.8f %12.5f\n" %\
                     (atoms.type_n[i].tostring().decode("utf-8"),\
                      atoms.pos[i, 0], atoms.pos[i, 1], atoms.pos[i, 2],\
                      atoms.mass[i])
            f.write(string)

    print("Wrote file: %s" % filename)



def readData(filename, format = "eon"):
    """Entry point for loading geometry files"""

    if "eon".startswith(format.lower()):
        mat, pos, t, idx = readEON(filename)
        return mat, pos, t, idx

    elif "lammps".startswith(format.lower()):
        mat, pos, t, idx = readLAMMPS(filename)
        return mat, pos, t, idx

    elif "vasp".startswith(format.lower()):
        mat, pos, t, idx = readVASP(filename)
        return mat, pos, t, idx

    elif "xyz".startswith(format.lower()):
        mat, pos, t, idx = readXYZ(filename)
        return mat, pos, t, idx

    else:
        print("Unrecognized file format")
        sys.exit()


def writeData(filename, atoms, format = "eon"):
    """Entry point for loading geometry files"""

    if "eon".startswith(format.lower()):
        writeEON(filename, atoms)

    elif "lammps".startswith(format.lower()):
        writeLAMMPS(filename, atoms)

    elif "vasp".startswith(format.lower()):
        writeVASP(filename, atoms)

    elif "xyz".startswith(format.lower()):
        writeXYZ(filename, atoms)

    else:
        print("Unrecognized file format")
        sys.exit()
