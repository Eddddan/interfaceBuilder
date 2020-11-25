#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt
from interfaceBuilder import interface

"""File containing utility functions for use in the interfaceBuilder code"""


def loadInterfaces(filename, verbose = 1):
    """Function for loading an interface collection saved in a .pkl file

    filename = str, Name of the file to load

    verbose = int, Print extra information, Deault = 1
    """

    try:
        with open(filename, "rb") as rf:
            obj = pickle.load(rf)

        if verbose > 0:
            string = "Loading data from: %s" % filename
            infoPrint(string)

            return obj

    except FileNotFoundError as e:
        string = "Error: File <%s> not found" % filename
        infoPrint(string)
        return None


def readPropFile(filename):
    """Function for reading files containing properties to set e.g. work of sepparation

    filename = str, Filename of file to read
    """
    
    try:
        """Read file"""
        with open(filename, 'r') as f:
            t = [np.int(j) for j in f.readline().split()]
            data = np.loadtxt(f)

            idx = data[:, 0].astype(np.int)
            val = data[:, 1:]

            return idx, t, val

    except FileNotFoundError:
        msg = "File: %s not found" % filename
        infoPrint(msg)
        return None, None, None
        

def iterateNrMatches(x, y, current, target, C, E, dC = 1,\
                     trace = 0, verbose = 1, current_iter = 0,\
                     max_iter = 500, tol = 1e-8, endpoint = "under"):
    """Basic function to iteratively find desiered number of interfaces
    based on a relation between atoms and strain atoms - C * strain ** E
    Used recursively until the correct number are found or stop critera
    have been met

    x = int, Atoms

    y = float, Strain

    current = int, Current matches found

    C = float, Value for C
    
    E = float, Value for E
    
    dC = float, modifyer to C per step

    trace = int, Counter for the number of steps in the same direction

    verbose = int, Print extra information, Default = 1

    max_iter = int, Max number of recursiv iterations

    tol = float, Tolerance for aborting if other criteria is not met

    endpoint = str("over"/"under"), If aborted due to tolerance match
    then this determines if the results will include as close to 
    but below the specified matches or above the specified matches
    """

    done = "Fail"
    current = np.sum((y - C * x ** E) < 0)

    if verbose > 0:
        string = "Iteration: %i/%i | Matches: %i/%i | C: %.4e | dC: %.4e | Trace: %3i"\
                 % (current_iter, max_iter, current, target, C, dC, trace)
        infoPrint(string)

    if current == target or dC < tol:
        if current == target:
            done = "Matches"
            return C, E, current, done, current_iter
        else:
            if current < target and endpoint.lower() == "under":
                done = "Tolerence"
                return C, E, current, done, current_iter
            elif current > target and endpoint.lower() == "over":
                done = "Tolerence"
                return C, E, current, done, current_iter

    if current_iter > max_iter:
        return C, E, current, done, current_iter

    current_iter += 1

    if current < target:
        if trace < 0: 
            trace = 0
            dC *= 0.5
        else:
            trace += 1
            dC *= 1.05

        C += dC
        C, E, current, done, current_iter = iterateNrMatches(x, y, current, target, C, E, dC = dC,\
                                      verbose = verbose, max_iter = max_iter, trace = trace,\
                                      current_iter = current_iter, tol = tol, endpoint = endpoint)

    elif current > target:
        if trace > 0: 
            trace = 0
            dC *= 0.5
        else:
            trace -= 1
            dC *= 1.05

        C -= dC
        C, E, current, done, current_iter = iterateNrMatches(x, y, current, target, C, E, dC = dC,\
                                      verbose = verbose, max_iter = max_iter, trace = trace,\
                                      current_iter = current_iter, tol = tol, endpoint = endpoint)


    return C, E, current, done, current_iter


def overlayLattice(lat, latRep, hAx, rot = 0, ls = '-', c = [0, 0, 0, 0.5],\
                   lw = 0.6):
    """Function for adding a lattice grid when plotting interfaces
    
    lat = array([2,2] or [3,3]), Lattice to overlay

    latRep = array([2,2]), Repetitions of the base lattice for the 
    plot teh overlay will be inposed on

    hAx = axes handle, Handle of the axis object where the lattice is to 
    be plotted

    rot = float, Rotation of the lattice in degrees

    ls = string(<valid_linestyle>), Valid matplotlib linestyle string to use
    when plotting the lattice

    c = <valid_color>, Valid matplotlib color specifier for the lattice

    lw = float, Line width of the lattice
    """

    """Pick out the a, b, x, y part of the lattice"""
    lat = lat[0:2, 0:2]

    """Rotate the lattice if specified"""
    aRad = np.deg2rad(rot)
    R = np.array([[np.cos(aRad), -np.sin(aRad)],
                  [np.sin(aRad),  np.cos(aRad)]])

    lat = np.matmul(R, lat)

    """Pick out the cell vectors"""
    a1 = lat[0:2, [0]]
    a2 = lat[0:2, [1]]

    scale = np.zeros((2, 4))
    scale[0:2, 0:2] = latRep
    scale[:, 2] = np.sum(scale[0:2, 0:2], axis = 1)

    """Check how big the lattice overlay needs to be"""
    nx_lo = np.min(scale[0, :]) 
    nx_hi = np.max(scale[0, :])
    ny_lo = np.min(scale[1, :])
    ny_hi = np.max(scale[1, :])

    extend = 25
    nX = np.arange(nx_lo - extend, nx_hi + 1 + extend)
    nY = np.arange(ny_lo - extend, ny_hi + 1 + extend)
    x = a1 * nX
    y = a2 * nY

    """Plot the grid"""
    xGrid = np.tile(x[0, [0, -1]], (y.shape[1], 1)) + y[[0], :].T
    yGrid = np.tile(x[1, [0, -1]], (y.shape[1], 1)) + y[[1], :].T
    hAx.plot(xGrid.T, yGrid.T, color = c, linewidth = lw, ls = ls)

    xGrid = np.tile(y[0, [0, -1]], (x.shape[1], 1)) + x[[0], :].T
    yGrid = np.tile(y[1, [0, -1]], (x.shape[1], 1)) + x[[1], :].T
    hAx.plot(xGrid.T, yGrid.T, color = c, linewidth = lw, ls = ls)


def infoPrint(string, sep = "=", sep_before = True, sep_after = True,\
              space_before = False, space_after = False):
    """Function for standardized print out of information

    string = str, String to be printed

    sep_before/after = True/False, Print separator line before/after string

    space_before/after = Print space before/after string
    """

    if space_before: print("")
    if sep_before: print(sep * len(string))
    print(string)
    if sep_after: print(sep * len(string))
    if space_after: print("")
    

def align(base, axis = 0, align_to = [1, 0], verbose = 1):
    """Function for aligning i_axis to cartesian align_to axes

    base = array([2,2] or [3,3]), Base to align

    axis = int(0/1), Base axis to align

    align_to = array([2,]), Cartesian xy axis to align base to

    verbose = int, Print extra information
    """

    norm_1 = np.linalg.norm(base[0:2, axis])
    norm_2 = np.linalg.norm(align_to[0:2])

    mat = np.zeros((2, 2))
    mat[:, 0] = base[0:2, axis]
    mat[:, 1] = align_to[0:2]

    """Calculate the angle between base and supplied axis"""
    ang = np.arccos(np.dot(base[0:2, axis], align_to[0:2]) / (norm_1 * norm_2))

    if np.linalg.det(mat) > 0:
        ang = 2 * np.pi - ang

    """Rotate clockwise"""
    ang *= -1
    R = getRotMatrix(ang, dim = 2, verbose = verbose - 1)

    if base.shape[0] == 2:
        R = R[0:2, 0:2]

    if verbose > 0:
        string = "Rotating axis %i by %.2f degrees" % (axis, np.rad2deg(ang))
        infoPrint(string)

    """Return the new base and the angle in radians"""
    return np.matmul(R, base), ang


def getRotMatrix(ang, dim = 2, verbose = 1):
    """Function for getting the rotation matrix for angle ang
    
    ang = float, Angles in radians

    verbose = int, Print extra information
    """

    if verbose > 0:
        string = "Returning %iD rotation matrix for angle %.2f deg"\
                 % (dim, np.rad2deg(ang))
        infoPrint(string)

    """Build rotation matrix"""
    R = np.array([[np.cos(ang), -np.sin(ang), 0],\
                  [np.sin(ang),  np.cos(ang), 0],\
                  [          0,            0, 1]])

    return R[0:dim, 0:dim]


def runningMean(data, chunk = 5):
    """Function for gettig the running mean of an array of data

    data = 1d np.array, Data to get running mean of

    chunk = int, Nr of point to use when calculating the mean
    """

    return np.convolve(data, np.ones((chunk,)) / chunk, mode = "valid")


def getPlotProperties():
    """Return available keyword properties"""

    string = "idx           = Index of current sorting\n"\
        "order         = Saved order\n"\
        "eps_11        = Eps_11\n"\
        "eps_22        = Eps_22\n"\
        "eps_12        = Eps_12\n"\
        "eps_mas       = Eps_mas\n"\
        "eps_max       = max(eps_11, eps_22, eps_12)\n"\
        "eps_max_a     = max(|eps_11, eps_22, eps_12|)\n"\
        "atoms         = Nr of atoms\n"\
        "angle         = Angle between interface cell vectors\n"\
        "rotation      = Initial rotation at creation\n"\
        "norm          = Sqrt(eps_11^2+eps_22^2+eps_12^2)\n"\
        "trace         = |eps_11|+|eps_22|\n"\
        "norm_trace    = Sqrt(eps_11^2+eps_22^2)\n"\
        "max_diag      = max(eps_11, eps_22)\n"\
        "max_diag_a    = max(|eps_11, eps_22|)\n"\
        "x             = Cell bounding box, x direction (a_1 aligned to x)\n"\
        "y             = Cell bounding box, y direction (a_1 aligned to x)\n"\
        "min_bound     = Min(x,y)\n"\
        "min_width     = Minimum width of the cell, min(l)*sin(cell_angle)\n"\
        "a_1           = Length of interface cell vector a_1\n"\
        "a_2           = Length of interface cell vector a_2\n"\
        "area          = Area of the interface\n"\
        "other         = Plot a custom array of values specified with keyword other. Length must match idx\n"\
        "e_int_c       = Interfacial energy, for specified translation(s)\n"\
        "e_int_d       = Interfacial energy (DFT), for specified translation(s)\n"\
        "e_int_diff_c  = Difference in Interfacial energy between translations\n"\
        "e_int_diff_d  = Difference in Interfacial energy (DFT) between translations\n"\
        "w_sep_c       = Work of adhesion, for specified translation(s)\n"\
        "w_sep_d       = Work of adhesion (DFT), for specified translation(s)\n"\
        "w_seps_c      = Work of adhesion (strained ref), for specified translation(s)\n"\
        "w_seps_d      = Work of adhesion (strained ref) (DFT), for specified translation(s)\n"\
        "w_sep_diff_c  = Difference in w_sep_c between tranlsations\n"\
        "w_sep_diff_d  = Difference in w_sep_d (DFT) between tranlsations\n"\
        "w_seps_diff_c = Difference in w_seps_c between translations\n"\
        "w_seps_diff_d = Difference in w_seps_d (DFT) between translations"
    
    return string


def getCellAngle(base, verbose = 1):
    """Function for returning the cell angle of a specific base.
    Angle + for right handed base and - for left handed base
       
    base = array([2,2] or [3,3]), Base to calculate the angles for

    verbose = int, Print extra information
    """

    a1 = base[0:2, 0]
    a2 = base[0:2, 1]
    norm = np.linalg.norm(base, axis = 0)
    
    ang = np.arccos(np.dot(a1, a2) / (norm[0] * norm[1]))
    if np.linalg.det(base) < 0:
        ang *= -1

    if verbose:
        string = "Angle between base vectors is %.2f degrees" % (np.rad2deg(ang))
        infoPrint(string)

    return ang


def center(base1, base2, verbose = 1):
    """Function for centering base2 around base1, in z direction

    base1 = array([2,2] or [3,3]), First base

    base2 = array([2,2] or [3,3]), Base to center around base1

    verbose = int, Print extra information
    """

    base2, ang = align(base2, align_to = base1[0:2, 0], verbose = False)
    ang_1 = getCellAngle(base1, verbose = verbose)
    ang_2 = getCellAngle(base2, verbose = verbose)
    ang_tot = ang + (ang_1 - ang_2) / 2

    new_base = rotate(base2, (ang_1 - ang_2) / 2, verbose = False)

    if verbose > 0:
        string = "Rotating base2 %.2f degrees, now %.2f degrees relative to base1"\
                 % (np.rad2deg(ang_tot), np.rad2deg((ang_1 - ang_2) / 2))
        infoPrint(string)

    """Return the new base and the angle in radians"""
    return new_base, ang_tot


def getCenter(base, verbose = 1):
    """Get the center point of the base

    base = array([2,2] or [3,3])

    verbose = int, Print extra information
    """

    base = base[0:2, 0:2]
    center = np.sum(base, axis = 1) / 2

    if verbose > 0:
        string = "Center point at (%.2f, %.2f)" % (center[0], center[1])
        infoPrint(string)

    return center


def rotate(vec, ang, verbose = 1):
    """Function for rotating vectors 2D/3D, around z axis

    vec = array([2,] or [3,]), Vector to rotate

    ang = flaot, Angles in radians

    verbose = int, Print extra information
    """

    R = np.array([[np.cos(ang), -np.sin(ang), 0],\
                  [np.sin(ang),  np.cos(ang), 0],\
                  [          0,            0, 1]])

    if vec.shape[0] == 2:
        R = R[0:2, 0:2]

    if verbose > 0:
        string = "Rotating vectors by %.2f degrees" % (np.rad2deg(ang))
        infoPrint(string)

    vec = np.matmul(R, vec)

    return vec


def calcStrains(a, b):
    """a is the unchanged bottom cell (target). b is the top
    cell that is to be strained to match a

    a, b = array([N,2,2]), N stacks of [2,2] cell vectors (x,y)
    to be matched against eachother 
    """
    
    """Angle between the first cell vector and cartesian x-axis"""
    aRad = np.arccos(a[:, 0, 0] / np.linalg.norm(a[:, :, 0], axis = 1))
    aRad[a[:, 1, 0] < 0] = 2 * np.pi - aRad[a[:, 1, 0] < 0]
    aRad *= -1

    """Rotate to align first axis of bottom cell (a) to cartesian x"""
    R = np.moveaxis(np.array([[np.cos(aRad), -np.sin(aRad)],
                              [np.sin(aRad),  np.cos(aRad)]]), 2, 0)
    a = np.matmul(R, a)

    """Now rotate the top cell to match first axis to the first axis of the bottom cell"""
    bRad = np.arccos(b[:, 0, 0] / np.linalg.norm(b[:, :, 0], axis = 1))
    bRad[b[:, 1, 0] < 0] = 2 * np.pi - bRad[b[:, 1, 0] < 0]
    bRad *= -1
        
    R = np.moveaxis(np.array([[np.cos(bRad), -np.sin(bRad)],
                              [np.sin(bRad),  np.cos(bRad)]]), 2, 0)

    b = np.matmul(R, b)

    """Calculate strain components"""
    eps_11 = np.abs(a[:, 0, 0] / b[:, 0, 0]) - 1
    eps_22 = np.abs(a[:, 1, 1] / b[:, 1, 1]) - 1
    eps_12 = 0.5 * (a[:, 0, 1] - (a[:, 0, 0] / b[:, 0, 0]) * b[:, 0, 1]) / b[:, 1, 1]

    """And mean absolute strain"""
    eps_mas = (np.abs(eps_11) + np.abs(eps_22) + np.abs(eps_12)) / 3

    return eps_11, eps_22, eps_12, eps_mas


def getAngles(a, b):
    """Function for finding the angles between vectors

    a, b = array([N, 2] or [N, 3]), N stacks of vectors to calculate
    the angles for
    """
    
    ang_ab = np.arccos(np.sum(a * b, axis = 1) /\
                      (np.linalg.norm(a, axis = 1) *\
                       np.linalg.norm(b, axis = 1)))
    
    return ang_ab


def extendCell(base, rep, pos, spec, mass):
    """Function for extending a unitcell.
       
    base - base for the cell
    
    rep  - [xLo, xHi, yLo, yHi, zLo, zHi]
    
    pos  - positions, cartesian in - cartesian out
    
    spec - Atomic species
    
    mass - Atomic masses
    """
       

    """Convert positions to direct coordinates"""
    pos_d = np.matmul(np.linalg.inv(base), pos)

    """Set up the extension"""
    m = np.arange(rep[0], rep[1] + 1)
    n = np.arange(rep[2], rep[3] + 1)
    k = np.arange(np.floor(rep[4]), np.ceil(rep[5]) + 1)

    """Build all permutations of the cell repetitions"""
    ext = np.array([np.tile(m, n.shape[0]),\
                    np.repeat(n, m.shape[0]),\
                    np.zeros((m.shape[0] * n.shape[0]))])
    ext = np.tile(ext, (1, k.shape[0]))
    ext[2, :] = ext[2, :] + np.repeat(k, m.shape[0] * n.shape[0])
    ext = np.repeat(ext, pos_d.shape[1], axis = 1)

    """Extend all positions to the repeated cells"""
    pos_d_ext = (ext + np.tile(pos_d, (1, np.int(ext.shape[1] / pos_d.shape[1]))))

    """Extend the atomic species tag along with the positions"""
    spec_ext = np.tile(spec, (np.int(ext.shape[1] / pos_d.shape[1])))

    """Extend the atomic masses along with the positions"""
    mass_ext = np.tile(mass, (np.int(ext.shape[1] / pos_d.shape[1])))

    """Remove all atoms in the z direction outside the supplied max/min z value"""
    keep = (pos_d_ext[2, :] >= rep[4]) * (pos_d_ext[2, :] < (rep[5] + 1))
    pos_d_ext = pos_d_ext[:, keep]
    spec_ext = spec_ext[keep]
    mass_ext = mass_ext[keep]

    """Transform from direct coordinates to Cartesian coordinates"""
    pos_ext = np.matmul(base, pos_d_ext)

    return pos_ext, spec_ext, mass_ext


def getTranslation(translation, surface, verbose = 1):
    """Function for getting translation vectors for specific surface

    translation = int, Specifier for the particular transltion of interest

    surface = str("0001" / "10-10" / "B100"), Specifier for the type of surface
    -------------------
    0001  = Basal plane of HCP
    10-10 = Prismatic plane of HCP
    B100  = 100 surface of BCC
    B110  = 110 surface of BCC
    F100  = 100 surface of FCC

    verbose = int, Print extra information
    """

    if not isinstance(translation, (int, np.integer)):
        if isinstance(translation, (list, np.ndarray)) and np.shape(translation)[0] > 1:
            T = np.array([translation[0], translation[1], 0])
            site = "Other"
            if verbose > 0:
                string = "Surface: %s | Translation made to site: %s"\
                    % (surface, site)
                infoPrint(string)
            return T, site
        else:
            T = np.array([0, 0, 0])
            site = "Top"
            if verbose > 0:
                string = "Surface: %s | Translation made to site: %s"\
                    % (surface, site)
                infoPrint(string)
            return T, site

    if surface is None:
        T = np.array([0, 0, 0])
        site = "Top"
        if verbose > 0:
            string = "Surface: %s | Translation made to site: %s"\
                 % (surface, site)
            infoPrint(string)
        return T, site

    if surface.lower() == "0001":
        if translation == 0:
            site = "Top"
            T = np.array([0, 0, 0])

        elif translation == 1:
            site = "Hollow-On"
            T = np.array([2/3, 1/3, 0])

        elif translation == 2:
            site = "Hollow-Off"
            T = np.array([1/3, -1/3, 0])

        elif translation == 3:
            site = "Bridge"
            T = np.array([0, 1/2, 0])

        elif translation > 3:
            site = "Translation out of range"
            T = np.array([0, 0, 0])

    elif surface.lower() == "10-10":
        if translation == 0:
            site = "Top"
            T = np.array([0, 0, 0])

        elif translation == 1:
            site = "Hollow"
            T = np.array([0.5, 0.5, 0])

        elif translation == 2:
            site = "Bridge-On"
            T = np.array([0, 0.5, 0])

        elif translation == 3:
            site = "Bridge-Off"
            T = np.array([0.5, 0, 0])

        elif translation > 3:
            site = "Translation out of range"
            T = np.array([0, 0, 0])

    elif surface.lower() == "b100":
        if translation == 0:
            site = "Top-Corner"
            T = np.array([0, 0, 0])

        elif translation == 1:
            site = "Top-Center"
            T = np.array([0.5, 0.5, 0])

        elif translation == 2:
            site = "Bridge"
            T = np.array([0.5, 0, 0])

        elif translation > 2:
            site = "Translation out of range"
            T = np.array([0, 0, 0])
            
    elif surface.lower() == "b110":
        if translation == 0:
            site = "Bridge-Top"
            T = np.array([0, 0, 0])

        elif translation == 1:
            site = "Bridge-Off"
            T = np.array([0.5, 0.5, 0])

        elif translation == 2:
            site = "Top"
            T = np.array([0.5, 0, 0])

        elif translation > 2:
            site = "Translation out of range"
            T = np.array([0, 0, 0])

    elif surface.lower() == "f100":
        if translation == 0:
            site = "Top"
            T = np.array([0, 0, 0])

        elif translation == 1:
            site = "Hollow"
            T = np.array([0.5, 0, 0])

        elif translation == 2:
            site = "Bride"
            T = np.array([0.25, 0.25, 0])

        elif translation > 2:
            site = "Translation out of range"
            T = np.array([0, 0, 0])

    elif surface.lower() == "f110":
        if translation == 0:
            site = "Top"
            T = np.array([0, 0, 0])

        elif translation == 1:
            site = "Bridge"
            T = np.array([0.5, 0, 0])

        elif translation == 2:
            site = "Hollow"
            T = np.array([0.5, 0.5, 0])

        elif translation > 2:
            site = "Translation out of range"
            T = np.array([0, 0, 0])

            
    if verbose > 0:
        string = "Surface: %s | Translation made to site: %s"\
                 % (surface, site)
        infoPrint(string)

    return T, site


def getNrTranslations(surface):
    """Get the total number of deafult translations for the specified surface

    surface = str("0001"/"10-10"/"b100"/"b110"/"f100"), Keyword for specific surface
    """

    if surface.lower() == "0001":
        return 4
    elif surface.lower() == "10-10":
        return 4
    elif surface.lower() == "b100":
        return 3
    elif surface.lower() == "b110":
        return 3
    elif surface.lower() == "f100":
        return 3
    elif surface.lower() == "f110":
        return 3
    else:
        return 0


def save_fig(filename = "Interface_figure.pdf", format = "pdf", dpi = 100, verbose = 1):
    """Function for saving figures

    filename = str, Name to save teh file to

    format = <valid_format>, Any valid matplotlib save format

    dpi = int, DPI when saving the figure

    verbose = int, Print extra information
    """

    if not filename.endswith(".%s" % format):
        filename += ".%s" % format
    plt.savefig(filename, format = format, dpi = dpi)
    if verbose > 0:
        string = "Saved figure: %s" % filename
        infoPrint(string)


def load_NN_array(filename):
    """Function for loading data from an NN array

    filename = str, name of the file containing the NN data
    """
    
    with open(filename, 'r') as f:
        data = np.loadtxt(f)


    """Unpack the first 3 fields"""
    interface = data[:, 0].astype(np.int)
    translation = data[:, 1].astype(np.int)
    element = data[:, 2].astype(np.int)

    """Pick out the mean values and the standard deviations"""
    NN = np.int((data.shape[1] - 3) / 2)
    mean = data[:, 3 : NN + 3]
    std = data[:, NN + 3:]

    si = np.argsort(interface)

    return mean, std, interface, translation, element


def get_NN_count(filename, cutoff = 3.8):
    """Function for getting the nr of NN within a specified cutoff

    filename = str, name of the file containing the NN data

    cutoff = float, Length cutoff for the NN count
    """

    mean, std, i_data, t_data, e_data = load_NN_array(filename)

    count = np.sum(mean < cutoff, axis = 1)
    count = np.reshape(count, (-1, np.max(t_data) + 1))

    return count
    

def plotNNC(filename, idx, trans = 0, row = 1, col = 1, N = 1, save = False,\
            format = "pdf", dpi = 100, verbose = 1, **kwargs):
    """Function for plotting NN data from NN arrays

    filename = str, name of the file containing the NN data

    idx = int, Plot these indicies

    row, col, N = row, column and number of the plot if used in subplot

    save = True/False/str, Save figure with standard name or with 
    specified string as filename

    dpi/format = int/<valid_format>, Format and dpi for saving

    verbose = int, Print extra information

    **kwargs = <valid_kwargs>, Valid matplotlib errorbar kwargs 
    """

    """Set some defaults"""
    if isinstance(idx, (np.integer, int)): idx = np.array([idx])
    if isinstance(idx, (range, list)): idx = np.array(idx)
    if isinstance(trans, (np.integer, int)): trans = np.array([trans])
    if isinstance(trans, (range, list)): trans = np.array(trans)

    """Load the data from specified file"""
    mean, std, i_data, t_data, e_data = load_NN_array(filename)

    """Set range for x values"""
    x = np.arange(1, mean.shape[1] + 1, dtype = np.int)
    
    hFig = plt.figure()
    hAx = plt.subplot(row, col, N)

    """Set some defaults"""
    ls = kwargs.pop("linestyle", "--")
    lw = kwargs.pop("linewidth", 0.5)
    m = kwargs.pop("marker", "o")
    ms = kwargs.pop("markersize", 4)
    cs = kwargs.pop("capsize", 2)
    elw = kwargs.pop("elinewidth", 1)

    for i in idx:
        for t in trans:
            y = mean[i_data == i, :][t_data[i_data == i] == t, :]
            yerr = std[i_data == i, :][t_data[i_data == i] == t, :]

            hAx.errorbar(x, y = y[0, :], yerr = yerr[0, :], linestyle = ls, linewidth = lw,\
                         marker = m, markersize = ms, capsize = cs, elinewidth = elw,\
                         label = "I-%i, T-%i" % (i, t), **kwargs)

    hAx.set_xlabel("Neighbor")
    hAx.set_ylabel("Distance, $(\AA)$")
    hAx.set_title("Nearest Neighbor Distances")
    hAx.legend(framealpha = 1, loc = "upper left")
        
    plt.tight_layout()
    if save:
        if save is True:
            save_fig(filename = "NN.%s" % (format), format = format,\
                     dpi = dpi, verbose = verbose)
        else:
            save_fig(filename = save, format = format, dpi = dpi,\
                     verbose = verbose)
            plt.close()
    else:
        plt.show()

