#!/usr/bin/env python3

import sys
import numpy as np
import utils as ut


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


def readEON(filename, verbose = 1):
    """Load EON geometry file"""

    if verbose > 0:
        string = "Reading file: %s, format: EON" % filename
        ut.infoPrint(string)

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
                mass = np.ones(Nt)
            elif i == 8:
                """Mass"""
                masses = np.array([np.float(x) for x in line.split()])
            elif i == 9:
                """Atom type of species 1"""
                t = np.chararray(Nt[-1], itemsize = 2)
                t[:Nt[nt]] = line
                mass[:Nt[nt]] = [masses[nt]]
            elif i > 10:
                l = line.split()
                if len(l) == 1:
                    t[Nt[nt] : Nt[nt + 1]] = line
                    mass[Nt[nt] : Nt[nt + 1]] = [masses[nt + 1]]
                    nt += 1
                elif len(l) == 4:
                    continue
                else:
                    pos[n, :] = [np.float(x) for x in l[:3]]
                    idx[n] = np.float(l[4])
                    n += 1

    """Convert lengths and angles to cell vectors"""
    mat = len2mat(vec, ang)

    return mat, pos, t, idx, mass

            

def readLAMMPS(filename, verbose = 1):
    """Load LAMMPS geometry file"""

    if verbose > 0:
        string = "Reading file: %s, format: LAMMPS" % filename
        ut.infoPrint(string)

    with open(filename, "r") as f:
        f.readline() #Skip first line (comment)

        for i, line in enumerate(f):
            if line in ['\n', '\r\n']: continue #Continue if empty
                
            l = line.split()
            if len(l) == 2 and l[1] == "atoms":
                t = np.ones(np.int(l[0]), dtype = np.int)
                idx = np.zeros(np.int(l[0]), dtype = np.int)
                mat = np.zeros((3, 3))
                pos = np.zeros((np.int(l[0]), 3))
                mass = np.ones(np.int(l[0]))
                nr = np.int(l[0])
            elif len(l) == 3 and l[2] == "types":
                masses = np.zeros(np.int(l[0]))
                types = np.int(l[0])
                break

        for i, line in enumerate(f) :
            if line in ['\n', '\r\n']: continue

            l = line.split()
            if len(l) == 4 and l[2] == "xlo":
                mat[0, 0] = np.float(l[1]) - np.float(l[0]) # xhi - xlo
            elif len(l) == 4 and l[2] == "ylo":
                mat[1, 1] = np.float(l[1]) - np.float(l[0]) # yhi - ylo
            elif len(l) == 4 and l[2] == "zlo":
                mat[2, 2] = np.float(l[1]) - np.float(l[0]) # zhi - zlo
            elif len(l) == 6 and l[3] == "xy":
                mat[0, 1] = np.float(l[0]) # xy
                mat[0, 2] = np.float(l[1]) # xz
                mat[1, 2] = np.float(l[2]) # yz
                break

        for i, line in enumerate(f):
            if line in ['\n', '\r\n']: continue
            
            n = 0
            if line.startswith("Masses"): #Read everything associated with the Masses tag
                for j, subline in enumerate(f):
                    if subline in ['\n', '\r\n']: continue
                    
                    l = subline.split()
                    masses[np.int(l[0]) - 1] = np.float(l[1])

                    n += 1
                    if n == types:
                        break

            n = 0
            if line.startswith("Atoms"): #Read everything associated with the Atoms tag
                if line in ['\n', '\r\n']: continue

                for j, subline in enumerate(f):
                    if subline in ['\n', '\r\n']: continue
                    
                    l = subline.split()
                    idx[n] = np.int(l[0]) - 1
                    t[n] = np.int(l[1])
                    pos[n, :] = (np.float(l[2]), np.float(l[3]), np.float(l[4]))
                    mass[n] = masses[t[n] - 1]
                    
                    n += 1
                    if n == nr:
                        break
        
    return mat, pos, t, idx, mass



def writeLAMMPS(filename, atoms, verbose = 1):
    """Write a LAMMPS data file"""
    
    if verbose > 0:
        string = "Writing file: %s, format: LAMMPS" % filename
        ut.infoPrint(string)

    """Make sure positions are in cartesian coordinates"""
    atoms.dir2car()

    with open(filename, "w") as f:
        
        """Newline character"""
        nl = "\n"

        box = [0, atoms.cell[0, 0], 0, atoms.cell[1, 1], 0, atoms.cell[2, 2]]
        nr_atoms = atoms.pos.shape[0]
        nr_types = np.unique(atoms.type_i).shape[0]
        masses = np.zeros(nr_types)
        for i, item in enumerate(np.unique(atoms.type_i)):
            masses[i] = atoms.mass[atoms.type_i == item][0]

        """First a comment line"""
        f.write("LAMMPS-data file (%s) written from file_io.py\n" % filename)
        f.write(nl)

        f.write("%i atoms\n" % nr_atoms)
        f.write("%i atom types\n" % nr_types)
        f.write(nl)

        f.write("%12.6f %12.6f   xlo xhi\n" % (box[0], box[1]))
        f.write("%12.6f %12.6f   ylo yhi\n" % (box[2], box[3]))
        f.write("%12.6f %12.6f   zlo zhi\n" % (box[4], box[5]))
        f.write("%12.6f %12.6f %12.6f   xy xz yz\n"\
                % (atoms.cell[0, 1], atoms.cell[0, 2], atoms.cell[1, 2]))
        f.write(nl)

        f.write("Masses\n")
        f.write(nl)
        for i, item in enumerate(masses):
            f.write("%i %11.6f\n" % (i + 1, item))
        f.write(nl)

        f.write("Atoms\n")
        f.write(nl)
        for i in range(nr_atoms):
            f.write("%5i %3i %12.6f %12.6f %12.6f\n"\
                    % (atoms.idx[i] + 1, atoms.type_i[i], atoms.pos[i, 0],\
                       atoms.pos[i, 1], atoms.pos[i, 2]))


def readVASP(filename):
    """Load VASP geometry file"""
    print("loadVASP")



def writeVASP(filename, atoms, direct = False, verbose = 1):
    """Write VASP POSCAR file"""

    if verbose > 0:
        string = "Writing file: %s, format: VASP" % filename
        ut.infoPrint(string)

    elements = []
    nr_elements = []
    for i in np.unique(atoms.type_i):
        elements.append(atoms.type_n[atoms.type_i == i][0].decode("utf-8"))
        nr_elements.append(np.shape(atoms.type_n[atoms.type_i == i])[0])

    if direct:
        atoms.car2dir()
        pos = atoms.pos
        coordinates = "Direct"
    else:
        atoms.dir2car()
        pos = atoms.pos
        coordinates = "Cartesian"

    frozen = np.chararray(atoms.pos.shape, itemsize = 1)
    frozen[:] = "F"
    frozen[atoms.frozen] = "T"

    with open(filename, "w") as f:
        
        """Newline character"""
        nl = "\n"

        scale = 1.0

        f.write("VASP POSCAR-file (%s) written from file_io.py\n" % filename)
        f.write("%s\n" % scale)

        """The cell is [a.T, b.T, c.T] i.e. row = cellvector, opposite of lammps""" 
        for row in atoms.cell.T:
            f.write("  %11.6f  %11.6f  %11.6f\n" % (row[0], row[1], row[2]))

        for e in elements:
            f.write("  %3s" % e)
        f.write("\n")

        for nr in nr_elements:
            f.write("  %3i" % nr)
        f.write("\n")

        f.write("Selective Dynamics\n")
        f.write("%s\n" % coordinates)
        for i, p in enumerate(pos):
            f.write("  %11.6f  %11.6f  %11.6f  %s  %s  %s\n"\
                        % (p[0], p[1], p[2], frozen[i, 0].decode("utf-8"),\
                           frozen[i, 0].decode("utf-8"), frozen[i, 0].decode("utf-8")))


def readXYZ(filename):
    """Load XYZ geometry file"""
    print("loadXYZ")


def writeXYZ(filename, atoms, verbose = 1):
    """Write a XYZ data file"""

    if verbose > 0:
        string = "Writing file: %s, format: XYZ" % filename
        ut.infoPrint(string)

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



def readData(filename, format = "eon"):
    """Entry point for loading geometry files"""

    if "eon".startswith(format.lower()):
        mat, pos, t, idx, mass = readEON(filename)
        return mat, pos, t, idx, mass

    elif "lammps".startswith(format.lower()):
        mat, pos, t, idx, mass = readLAMMPS(filename)
        return mat, pos, t, idx, mass

    elif "vasp".startswith(format.lower()):
        mat, pos, t, idx, mass = readVASP(filename)
        return mat, pos, t, idx, mass

    elif "xyz".startswith(format.lower()):
        mat, pos, t, idx, mass = readXYZ(filename)
        return mat, pos, t, idx, mass

    else:
        print("Unrecognized file format")
        sys.exit()


def writeData(filename, atoms, format = "eon", verbose = 1):
    """Entry point for loading geometry files"""

    if "eon".startswith(format.lower()):
        writeEON(filename, atoms, verbose = verbose)

    elif "lammps".startswith(format.lower()):
        writeLAMMPS(filename, atoms, verbose = verbose)

    elif "vasp".startswith(format.lower()):
        writeVASP(filename, atoms, verbose = verbose)

    elif "xyz".startswith(format.lower()):
        writeXYZ(filename, atoms, verbose = verbose)

    else:
        print("Unrecognized file format")
        sys.exit()
