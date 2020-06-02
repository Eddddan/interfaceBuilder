
import pickle
import numpy as np
import matplotlib.pyplot as plt

def loadInterfaces(filename, verbose = 1):
    """Function for loading an interface connection saved to a .pkl file"""

    with open(filename, "rb") as rf:
        obj = pickle.load(rf)

    if verbose > 0:
        string = "Loading data from: %s" % filename
        infoPrint(string)

    return obj




def iterateNrMatches(x, y, current, target, C, E, dC = 1,\
                     trace = 0, verbose = 1, current_iter = 0,\
                     max_iter = 500, tol = 1e-8, endpoint = "under"):

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



def overlayLattice(lat, latRep, hAx, rot = 0, ls = '-', c = [0, 0, 0, 0.4],\
                   lw = 0.5):
    """Function for adding a lattice grid when plotting interfaces"""

    """Pick out the a, b, x, y part of the lattice"""
    lat = lat[0:2, 0:2]

    """Rotate the lattice if specified"""
    aRad = np.deg2rad(rot)
    R = np.array([[np.cos(aRad), -np.sin(aRad)],
                  [np.sin(aRad),  np.cos(aRad)]])

    lat = np.matmul(R, lat)

    a1 = lat[0:2, [0]]
    a2 = lat[0:2, [1]]

    scale = np.zeros((2, 4))
    scale[0:2, 0:2] = latRep
    scale[:, 2] = np.sum(scale[0:2, 0:2], axis = 1)

    nx_lo = np.min(scale[0, :]) 
    nx_hi = np.max(scale[0, :])
    ny_lo = np.min(scale[1, :])
    ny_hi = np.max(scale[1, :])

    extend = 10
    nX = np.arange(nx_lo - extend, nx_hi + 1 + extend)
    nY = np.arange(ny_lo - extend, ny_hi + 1 + extend)
    x = a1 * nX
    y = a2 * nY

    xGrid = np.tile(x[0, [0, -1]], (y.shape[1], 1)) + y[[0], :].T
    yGrid = np.tile(x[1, [0, -1]], (y.shape[1], 1)) + y[[1], :].T
    hAx.plot(xGrid.T, yGrid.T, color = c, linewidth = lw, ls = ls)

    xGrid = np.tile(y[0, [0, -1]], (x.shape[1], 1)) + x[[0], :].T
    yGrid = np.tile(y[1, [0, -1]], (x.shape[1], 1)) + x[[1], :].T
    hAx.plot(xGrid.T, yGrid.T, color = c, linewidth = lw, ls = ls)



def infoPrint(string, sep = "=", sep_before = True, sep_after = True,\
              space_before = False, space_after = False):
    if space_before: print("")
    if sep_before: print(sep * len(string))
    print(string)
    if sep_after: print(sep * len(string))
    if space_after: print("")
    


def align(base, axis = 0, align_to = [1, 0], verbose = 1):
    """Function for aligning i_axis to cartesian align_to axes"""

    norm_1 = np.linalg.norm(base[0:2, axis])
    norm_2 = np.linalg.norm(align_to[0:2])

    mat = np.zeros((2, 2))
    mat[:, 0] = base[0:2, axis]
    mat[:, 1] = align_to[0:2]

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
    """Function for getting the rotation matrix for angle ang"""

    if verbose > 0:
        string = "Returning %iD rotation matrix for angle %.2f deg"\
                 % (dim, np.rad2deg(ang))
        infoPrint(string)

    R = np.array([[np.cos(ang), -np.sin(ang), 0],\
                  [np.sin(ang),  np.cos(ang), 0],\
                  [          0,            0, 1]])

    return R[0:dim, 0:dim]


def getCellAngle(base, verbose = 1):
    """Function for returning the cell angle of a specific base.
       Angle + for right handed base and - for left handed base"""

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
    """Function for centering base2 around base1, in z direction"""

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
    """Get the center point of the base"""

    base = base[0:2, 0:2]
    center = np.sum(base, axis = 1) / 2

    if verbose > 0:
        string = "Center point at (%.2f, %.2f)" % (center[0], center[1])
        infoPrint(string)

    return center



def rotate(vec, ang, verbose = 1):
    """Function for rotating vectors 2D/3D, around z axis"""

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
    cell that is to be strained to match a"""
    
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
    """Function for finding the angle between vectors"""
    
    ang_ab = np.arccos(np.sum(a * b, axis = 1) /\
                      (np.linalg.norm(a, axis = 1) *\
                       np.linalg.norm(b, axis = 1)))
    
    return ang_ab



def extendCell(base, rep, pos, spec):
    """Function for extending a unitcell.
       base - base for the cell
       rep  - [xLo, xHi, yLo, yHi, zLo, zHi]
       pos  - positions, cartesian in - cartesian out
       spec - Atomic species"""

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

    """Remove all atoms in the z direction outside the supplied max/min z value"""
    keep = (pos_d_ext[2, :] >= rep[4]) * (pos_d_ext[2, :] < (rep[5] + 1))
    pos_d_ext = pos_d_ext[:, keep]
    spec_ext = spec_ext[keep]

    """Transform from direct coordinates to Cartesian coordinates"""
    pos_ext = np.matmul(base, pos_d_ext)

    return pos_ext, spec_ext



def getTranslation(translate, surface, verbose = 1):
    """Function for getting translation vectors for specific surface"""

    if type(translate) == np.ndarray or type(translate) == list:
        if np.shape(translate)[0] != 2:
            return np.array([0, 0, 0])
        else:
            T = np.zeros(3)
            T[0:2] = translate
            return T

    if surface == "0001":
        if translate == 1:
            site = "Top"
            T = np.array([0, 0, 0])

        elif translate == 2:
            site = "Hollow-On"
            T = np.array([2/3, 1/3, 0])

        elif translate == 3:
            site = "Hollow-Off"
            T = np.array([1/3, -1/3, 0])

        elif translate == 4:
            site = "Bridge"
            T = np.array([0, 1/2, 0])
            
    if verbose > 0:
        string = "Surface: %s | Translation made to site: %s"\
                 % (surface, site)
        infoPrint(string)

    return T



def save_fig(filename = "Interface_figure.pdf", format = "pdf", dpi = 100, verbose = 1):
    """Function for saving figures"""
    if not filename.endswith(".%s" % format):
        filename += ".%s" % format
    plt.savefig(filename, format = format, dpi = dpi)
    if verbose > 0:
        string = "Saved figure: %s" % filename
        infoPrint(string)