
import pickle
import inputs
import file_io
import structure
import utils as ut
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Interface():
    """
    Class for holdiding a collection of generated interfaces.
    """

    __slots__ = ['cell_1', 'cell_2', 'rep_1', 'rep_2', 'eps_11', 'eps_22',\
                 'eps_12', 'eps_mas', 'atoms', 'ang', 'e_int', 'base_1',\
                 'base_2', 'pos_1', 'pos_2', 'spec_1', 'spec_2', 'mass']

    def __init__(self,\
                 structure_a,\
                 structure_b):

        self.cell_1 = None
        self.cell_2 = None
        self.rep_1 = None
        self.rep_2 = None
        self.eps_11 = None
        self.eps_22 = None
        self.eps_12 = None
        self.eps_mas = None
        self.atoms = None
        self.ang = None
        self.e_int = None

        self.base_1 = structure_a.cell
        self.base_2 = structure_b.cell
        self.pos_1 = structure_a.pos
        self.pos_2 = structure_b.pos
        self.spec_1 = structure_a.type_n
        self.spec_2 = structure_b.type_n
        self.mass = None



    def deleteInterfaces(self, keep, verbose = 1):
        """Function for removing interfaces"""

        self.cell_1 = self.cell_1[keep]
        self.cell_2 = self.cell_2[keep]
        self.rep_1 = self.rep_1[keep]
        self.rep_2 = self.rep_2[keep]

        self.eps_11 = self.eps_11[keep]
        self.eps_22 = self.eps_22[keep]
        self.eps_12 = self.eps_12[keep]
        self.eps_mas = self.eps_mas[keep]

        self.atoms = self.atoms[keep]
        self.ang = self.ang[keep]
        self.e_int = self.e_int[keep]

        if verbose > 0:
            print("Interfaces deleted: %i | Interfaces remaining: %i"\
                  % (np.sum(np.logical_not(keep)), np.sum(keep)))



    def sortInterfaces(self, sort = "atoms", rev = False):
        """Function for sorting the interfaces"""

        sort = sort.lower()
        sortable_properties = ["atoms",   "angle",  "area",\
                               "eps_11",  "eps_22", "eps_12",\
                               "eps_mas", "e_int",\
                               "base_angle_1", "base_angle_2"]

        if sort == "atoms":
            si = np.argsort(self.atoms)
        elif sort == "angle":
            si = np.argsort(self.ang)
        elif sort == "e_int":
            si = np.argsort(self.e_int)
        elif sort == "eps_11":
            si = np.argsort(np.abs(self.eps_11))
        elif sort == "eps_22":
            si = np.argsort(np.abs(self.eps_22))
        elif sort == "eps_12":
            si = np.argsort(np.abs(self.eps_12))
        elif sort == "eps_mas":
            si = np.argsort(self.eps_mas)
        elif sort == "area":
            si = np.argsort(self.getAreas())
        elif sort == "base_angle_1":
            si = np.argsort(self.getBaseAngles(cell = 1))
        elif sort == "base_angle_2":
            si = np.argsort(self.getBaseAngles(cell = 2))
        else:
            print("Unknown sorting option: %s, sortable properties are:" % sort)
            for i, item in enumerate(sortable_properties):
                print("%2i. %s" % (i, item))
            return

        if rev:
            si = si[::-1]

        self.indexSortInterfaces(index = si)



    def indexSortInterfaces(self, index):
        """Sort interfaces based on supplied index"""

        self.cell_1 = self.cell_1[index]
        self.cell_2 = self.cell_2[index]
        self.rep_1 = self.rep_1[index]
        self.rep_2 = self.rep_2[index]

        self.eps_11 = self.eps_11[index]
        self.eps_22 = self.eps_22[index]
        self.eps_12 = self.eps_12[index]
        self.eps_mas = self.eps_mas[index]

        self.atoms = self.atoms[index]
        self.ang = self.ang[index]
        self.e_int = self.e_int[index]



    def getAtomStrainRatio(self, strain = "eps_mas", const = None, exp = 1, verbose = 1):
        """Get the property atoms - A * abs(strain) ** B"""

        if const is None:
            const, exp = self.getAtomStrainExpression(strain = strain, verbose = verbose - 1)

        if verbose > 0: 
            string = "Returning values of expression: atoms - A * abs(strain)^B with A,B: %.3e, %.3e"\
                     % (const, exp)
            ut.infoPrint(string)

        if strain.lower() == "eps_11":
            return self.atoms - const * np.abs(self.eps_11) ** exp
        elif strain.lower() == "eps_22":
            return self.atoms - const * np.abs(self.eps_22) ** exp
        elif strain.lower() == "eps_12":
            return self.atoms - const * np.abs(self.eps_12) ** exp
        elif strain.lower() == "eps_mas":
            return self.atoms - const * self.eps_mas ** exp

        else:
            print("Unknown option: %s" % strain)



    def getAtomStrainMatches(self, matches = 100, const = None, exp = 1,\
                             strain = "eps_mas", verbose = 1, max_iter = 500,\
                             tol = 1e-8, endpoint = "under"):
        """Function for returning interfaces matching the critera
           A * abs(strain) ** B"""

        if const is None:
            const, exp = self.getAtomStrainExpression(strain = strain, verbose = verbose - 1)

        if strain == "eps_11":
            eps = self.eps_11.copy()
        elif strain == "eps_22":
            eps = self.eps_22.copy()
        elif strain == "eps_12":
            eps = self.eps_12.copy()
        elif strain == "eps_mas":
            eps = self.eps_mas.copy()

        atoms = self.atoms.copy()

        current = np.sum((atoms - const * eps ** exp) < 0)

        """Recursive function to find the correct fit to get specified nr of matches"""
        C, E, current, done, iter = ut.iterateNrMatches(eps, atoms, current = current, target = matches,\
                                                  C = const, E = exp, dC = 0.1, verbose = verbose - 1,\
                                                  max_iter = max_iter, current_iter = 0, tol = tol,\
                                                  endpoint = endpoint)

        if verbose > 0:
            string = "Iterations: %i | Matches (%i): %i | const: %.3e | exp: %.3e | Status: %s"\
                 % (iter, matches, current, C, E, done)
            ut.infoPrint(string)

        return C, E
    


    def getAtomStrainExpression(self, strain = "eps_mas", verbose = 1):
        """Get the curve from min(log(abs(strain))) --> min(log(atoms))"""

        if strain.lower() == "eps_11":
            si1 = np.lexsort((self.atoms, np.abs(self.eps_11)))[0]
            si2 = np.lexsort((np.abs(self.eps_11), self.atoms))[0]

            eps = np.array([np.abs(self.eps_11[si1]), np.abs(self.eps_11[si2])])
            atoms = np.array([self.atoms[si1], self.atoms[si2]])

        elif strain.lower() == "eps_22":
            si1 = np.lexsort((self.atoms, np.abs(self.eps_22)))[0]
            si2 = np.lexsort((np.abs(self.eps_22), self.atoms))[0]

            eps = np.array([np.abs(self.eps_22[si1]), np.abs(self.eps_22[si2])])
            atoms = np.array([self.atoms[si1], self.atoms[si2]])

        elif strain.lower() == "eps_12":
            si1 = np.lexsort((self.atoms, np.abs(self.eps_12)))[0]
            si2 = np.lexsort((np.abs(self.eps_12), self.atoms))[0]

            eps = np.array([np.abs(self.eps_12[si1]), np.abs(self.eps_12[si2])])
            atoms = np.array([self.atoms[si1], self.atoms[si2]])

        elif strain.lower() == "eps_mas":
            si1 = np.lexsort((self.atoms, self.eps_mas))[0]
            si2 = np.lexsort((self.eps_mas, self.atoms))[0]

            eps = np.array([self.eps_mas[si1], self.eps_mas[si2]])
            atoms = np.array([self.atoms[si1], self.atoms[si2]])

        else:
            print("Unknown option: %s" % strain)
            return

        eps = np.log(eps)
        atoms = np.log(atoms)
        
        """In y = A*x**B find A and B"""
        B = (atoms[1] - atoms[0]) / (eps[1] - eps[0])
        A = np.exp(atoms[0] - B * eps[0])
        
        if verbose > 0:
            string = "Expression found (A * x^B): %.3e * x^%.3e" % (A, B)
            ut.infoPrint(string)

        return A, B


    def getAreas(self):
        """function for getting the area of all interaces"""

        return np.abs(np.linalg.det(self.cell_1))



    def getBaseAngles(self, cell):
        """Function fofr getting the base angles of the bottom or top cells"""

        if cell == 1:
            ang = np.arccos(np.sum(self.cell_1[:, :, 0] * self.cell_1[:, :, 1], axis = 1) /\
                            np.prod(np.linalg.norm(self.cell_1, axis = 1), 1))
        elif cell == 2:
            ang = np.arccos(np.sum(self.cell_2[:, :, 0] * self.cell_2[:, :, 1], axis = 1) /\
                            np.prod(np.linalg.norm(self.cell_2, axis = 1), 1))

        return ang



    def getBaseLengths(self, cell):
        """Function for getting the cell lengths"""

        if cell == 1:
            l = np.linalg.norm(self.cell_1, axis = 1)
        elif cell == 2:
            l = np.linalg.norm(self.cell_1, axis = 1)

        return l



    def matchCells(self, dTheta = 4, theta = None, n_max = 4, N = None,\
                   m_max = 4, M = None, max_strain = 1, max_atoms = 1000,\
                   limit = None, exp = 1, verbose = 1, min_angle = 12.5):

        """Get number of atoms per area (xy) in base cell 1 and 2"""
        rhoA = self.pos_1.shape[0] / np.abs(np.cross(self.base_1[0:2, 0], self.base_1[0:2, 1]))
        rhoB = self.pos_2.shape[0] / np.abs(np.cross(self.base_2[0:2, 0], self.base_2[0:2, 1]))

        """Cell rotation angles"""
        if theta is not None:
            if isinstance(theta, (int, np.integer)):
                angle = np.array([theta])
            else:
                angle = np.array(theta)
        else:
            angle = np.arange(0, 180, dTheta)

        """Repetions of the first cell vector, [-n_max,...,n_max],
           N takes president as a single specific repitition"""
        if N is not None:
            nR = np.arange(N, N + 1)
        else:
            nR = np.arange(-n_max, n_max + 1)

        """Repetions of the second cell vector, [0,...,m_max],
           M takes president as a single specific repitition"""
        if M is not None:
            mR = np.arange(M, M + 1)
        else:
            mR = np.arange(0, m_max + 1)

        """Create all permutations of nR and mR"""
        dPerm = np.mgrid[nR[0]:nR[-1] + 1, mR[0]:mR[-1] + 1].reshape(2, nR.shape[0] * mR.shape[0])

        """Convert angle to radians"""
        aRad = np.deg2rad(angle)

        """Set up a Rotation matrix, move axis to work with shapes (X,2,2)"""
        R = np.moveaxis(np.array([[np.cos(aRad), -np.sin(aRad)],
                                  [np.sin(aRad),  np.cos(aRad)]]), 2, 0)

        """Rotate the B cell by the specified angles, e.g. C = R*B"""
        C = np.matmul(R, self.base_2[0:2, 0:2])

        """Build all possible cell vectors given the permutations dPerm
        d = C*dPerm each row will be a possible cell vector"""
        d = np.matmul(C, dPerm)

        """Express d in basis cell 1, d = A*e, find e -> A(-1)*d = e"""
        e = np.matmul(np.linalg.inv(self.base_1[0:2, 0:2]), d)

        """Snap the e vectors to the A grid by rounding e to integers"""
        e = np.round(e, 0).astype(int)

        """Caclculate the new (strained) d vectors (f), f = A * eInt"""
        f = np.matmul(self.base_1[0:2, 0:2], e)

        """Create all permutations of the f vectors"""
        F = np.zeros((angle.shape[0], f.shape[2]**2, 2, 2))
        F[:, :, :, 0] = np.swapaxes(np.tile(f, f.shape[2]), 1, 2)
        F[:, :, :, 1] = np.swapaxes(np.repeat(f, f.shape[2], axis = 2), 1, 2)

        """Flatten the first 2 dimensions"""
        F = F.reshape(-1, *F.shape[-2:])

        """Create all the same permutations of the d vectors"""
        D = np.zeros((angle.shape[0], d.shape[2]**2, 2, 2))
        D[:, :, :, 0] = np.swapaxes(np.tile(d, d.shape[2]), 1, 2)
        D[:, :, :, 1] = np.swapaxes(np.repeat(d, d.shape[2], axis = 2), 1, 2)

        """Flatten the first 2 dimensions"""
        D = D.reshape(-1, *D.shape[-2:])

        """Create all the same permutations of the eInt vectors"""
        FRep = np.zeros((angle.shape[0], e.shape[2]**2, 2, 2))
        FRep[:, :, :, 0] = np.swapaxes(np.tile(e, e.shape[2]), 1, 2)
        FRep[:, :, :, 1] = np.swapaxes(np.repeat(e, e.shape[2], axis = 2), 1, 2)

        """Flatten the first 2 dimensions"""
        FRep = FRep.reshape(-1, *FRep.shape[-2:])

        """Create all the same permutations of the dPerm vectors"""
        dPerm = np.tile(dPerm[np.newaxis, :, :], (angle.shape[0], 1, 1))
        DRep = np.zeros((angle.shape[0], dPerm.shape[2]**2, 2, 2))
        DRep[:, :, :, 0] = np.swapaxes(np.tile(dPerm, dPerm.shape[2]), 1, 2)
        DRep[:, :, :, 1] = np.swapaxes(np.repeat(dPerm, dPerm.shape[2], axis = 2), 1, 2)

        """Flatten the first 2 dimensions"""
        DRep = DRep.reshape(-1, *DRep.shape[-2:])

        """Calculate the area of the F and D cells"""
        detF = np.linalg.det(F)
        detD = np.linalg.det(D)

        """Remove all combinations where the determinant is 0 or <0
           i.e. linearly dependent or wrong handed. Do the same for 
           the top cell"""
        keep = (detF > 1e-6) * (detD > 1e-6)
        detF = detF[keep]
        detD = detD[keep]

        if verbose > 0:
            print("Total basis pairs: %.0f | Lin dep/left handed: %.0f | Total kept: %.0f"\
                      % (keep.shape[0], keep.shape[0] - np.sum(keep), np.sum(keep)))

        """Remove the lin-dep/left handed combinations before calculating the strain"""
        F = F[keep]
        D = D[keep]
        FRep = FRep[keep]
        DRep = DRep[keep]

        """Calculate the strain of the new cell vectors"""
        eps_11, eps_22, eps_12, eps_mas = ut.calcStrains(F, D)

        """Create a matching vector with the original rotations"""
        ang = np.repeat(angle, f.shape[2]**2)
        ang = ang[keep]

        """Calculate the number of atoms using the area and the area density"""
        rawAtoms = rhoA * detF + rhoB * detD
        atoms = np.round(rawAtoms)

        """Check to make sure the calculated nr of atoms are integers, otherwise flag it""" 
        tol = 7
        flag = (atoms != np.round(rawAtoms, tol))
        if np.sum(flag) != 0:
            index = np.arange(atoms.shape[0])[flag]
            string = "None integer number of atoms calculated for the following interfaces"
            ut.infoPrint(string, sep_before = False)
            for i in index:
                print("Index: %6i | Nr atoms: %14.10f" % (i, rawAtoms[i]))

        """Keep only unique entries. Found by checking for unique pairs of
           combinations for bottom and top surfaces"""
        full = np.zeros((atoms.shape[0], 4 * 2))
        full[:, 0:4] = FRep.reshape(*FRep.shape[0:1], -1)
        full[:, 4:8] = DRep.reshape(*DRep.shape[0:1], -1)

        ufi = np.unique(full, axis = 0, return_index = True)[1]
        keep = np.isin(np.arange(atoms.shape[0]), ufi)
        if verbose > 0:
            print("Non unique matches: %i | Total matches keept: %i"\
                      % (atoms.shape[0] - np.sum(keep), np.sum(keep)))

        """Assign values to class variables"""
        self.cell_1 = F[keep]
        self.cell_2 = D[keep]
        self.rep_1 = FRep[keep]
        self.rep_2 = DRep[keep]
        self.eps_11 = eps_11[keep]
        self.eps_22 = eps_22[keep]
        self.eps_12 = eps_12[keep]
        self.eps_mas = eps_mas[keep]
        self.atoms = atoms[keep]
        self.ang = ang[keep]
        self.e_int = np.zeros(self.atoms.shape[0])

        """Further removal of interfaces based on specified critera follows below"""

        """Reject interfaces based on criteria of strain * atoms^exp > limit"""
        if limit is not None:
            keep = ((self.eps_mas * (self.atoms ** exp)) < limit)
            ratio = np.sum(np.logical_not(keep))
            if verbose > 0:
                print("Matches with (strain * atoms^%s) > %s: %i | Total matches kept: %i"\
                      % (exp, limit, ratio, np.sum(keep)))

            """Remove interfaces with strain*atoms^exp > limit"""
            self.deleteInterfaces(keep)

        """Remove cells with to narrow cell angles, defined below"""
        ang_lim = np.deg2rad(min_angle)
        ang_1 = self.getBaseAngles(cell = 1)
        ang_2 = self.getBaseAngles(cell = 2)

        keep = (ang_1 > ang_lim) * (ang_1 < np.pi - ang_lim) *\
               (ang_2 > ang_lim) * (ang_2 < np.pi - ang_lim)

        max_angle = np.sum(np.logical_not(keep))
        if verbose > 0:
            print("Cell angle outside limit (%.1f<X<%.1f): %i | Total kept: %i"\
                  % (np.rad2deg(ang_lim), np.rad2deg(np.pi - ang_lim), max_angle, np.sum(keep)))

        """Remove interfaces with angles outside specified range"""
        self.deleteInterfaces(keep)

        """Remove matches were any strain component is > max_strain"""
        keep = (np.abs(self.eps_11) < max_strain) *\
               (np.abs(self.eps_22) < max_strain) *\
               (np.abs(self.eps_12) < max_strain)

        max_strain = np.sum(np.logical_not(keep))
        if verbose > 0:
            print("Matches above max strain: %i | Total matches kept: %i"\
                  % (max_strain, np.sum(keep)))

        """Remove interfaces with abs(strains) above max_strain"""
        self.deleteInterfaces(keep)

        """Remove matches with the number of atoms > max_atoms"""
        keep = (self.atoms < max_atoms)
        max_atoms = np.sum(np.logical_not(keep))
        if verbose > 0:
            print("Matches with to many atoms: %i | Total matches kept: %i"\
                  % (max_atoms, np.sum(keep)))

        """Remove interfaces with more atoms than max_atoms"""
        self.deleteInterfaces(keep)

        """Sort the interaces based on number of atoms"""
        self.sortInterfaces()



    def printInterfaces(self, idx = None, sort = None, rev = False, flag = None):
        """Print info about found interfaces"""

        if idx is None:
            idx = range(self.atoms.shape[0])
        elif isinstance(idx, (int, np.integer)):
            idx = [idx]

        if sort is not None:
            self.sortInterface(sort = sort, rev = rev)

        header1 = "%6s | %5s | %3s %-9s | %5s | %5s | %-6s | %-5s | %4s %-25s | %3s %-11s | %3s %-11s "\
                  % ("Index", "Rot", "", "Length", "Angle", "Angle", "Area", "Atoms", "", "Strain",\
                     "", "Lattice A", "", "Lattice B")
        header2 = "%6s | %5s | %6s, %5s | %5s | %5s | %6s | %5s | %7s,%7s,%7s,%6s | "\
                  "%3s,%3s,%3s,%3s | %3s,%3s,%3s,%3s"\
                  % ("i", "b0/x", "a1", "a2", "b1/b2", "a1/a2", "Ang^2", "Nr", "11", "22", "12", "mas",\
                     "a1x", "a1y", "a2x", "a2y", "b1x", "b1y", "b2x", "b2y")

        div = "=" * len(header1)
        print("\n" + header1 + "\n" + header2 + "\n" + div)

        n = 0
        for i in idx:

            la = np.linalg.norm(self.cell_1[i, :, :], axis = 0)
            lb = np.linalg.norm(self.cell_2[i, :, :], axis = 0)

            aa = np.dot(self.cell_1[i, :, 0], self.cell_1[i, :, 1]) / (la[0] * la[1])
            aa = np.rad2deg(np.arccos(aa))

            ba = np.dot(self.cell_2[i, :, 0], self.cell_2[i, :, 1]) / (lb[0] * lb[1])
            ba = np.rad2deg(np.arccos(ba))

            ar = np.abs(np.sin(np.deg2rad(aa))) * la[0] * la[1]

            s1 = self.eps_11[i] * 100
            s2 = self.eps_22[i] * 100
            s3 = self.eps_12[i] * 100
            s4 = self.eps_mas[i] * 100

            ra = self.rep_1[i, :, :].flatten()
            rb = self.rep_2[i, :, :].flatten()
            b1 = self.ang[i]

            at = self.atoms[i]

            if np.isin(i, flag):
                string = "%6.0f * %5.1f * %6.1f,%6.1f * %5.1f * %5.1f * %6.1f * %5.0f * "\
                    "%7.2f,%7.2f,%7.2f,%6.2f * %3i,%3i,%3i,%3i * %3i,%3i,%3i,%3i"\
                    % (i, b1, la[0], la[1], ba, aa, ar, at, s1, s2, s3, s4,\
                           ra[0], ra[2], ra[1], ra[3], rb[0], rb[2], rb[1], rb[3])
            else:
                string = "%6.0f | %5.1f | %6.1f,%6.1f | %5.1f | %5.1f | %6.1f | %5.0f | "\
                    "%7.2f,%7.2f,%7.2f,%6.2f | %3i,%3i,%3i,%3i | %3i,%3i,%3i,%3i"\
                    % (i, b1, la[0], la[1], ba, aa, ar, at, s1, s2, s3, s4,\
                           ra[0], ra[2], ra[1], ra[3], rb[0], rb[2], rb[1], rb[3])

            print(string)

        print(div + "\n")



    def subplotInterface(self, annotate = True, idx = 0, verbose = 1,\
                         align_base = "cell_1", scale = False, save = False,\
                         format = "pdf", dpi = 100, row = None,\
                         col = None):

        if type(idx) != list:
            """Just send it to the standard plotInterfaces"""
            self.plotInterface(idx = idx, annotate = annotate,\
                               verbose = verbose, align_base = align_base,\
                               scale = scale, save = save, format = format,\
                               dpi = dpi, handle = False)
            return

        if verbose > 0:
            self.printInterfaces(idx)

        if row is None and col is None:
            col = len(idx)
            row = 1
        elif col is None:
            col = np.int(np.ceil(len(idx) / row))
        elif row is None:
            row = np.int(np.ceil(len(idx) / col))

        hFig = plt.figure()
        for N, item in enumerate(idx):

            self.plotInterface(annotate = annotate, idx = item, verbose = verbose - 1,\
                               align_base = align_base, scale = scale, save = False,\
                               handle = True, col = col, row = row, N = N+1)

        plt.tight_layout(h_pad = 0.3, w_pad = 0.3)
        if save:
            if save is True:
                add = ""
                for i in idx:
                    add += "_%s" % i
                save_fig(filename = "interface%s.%s" % (add, format), format = format,\
                         dpi = dpi, verbose = verbose)
            else:
                save_fig(filename = save, format = format, dpi = dpi,\
                         verbose = verbose)
            plt.close()
        else:
            plt.show()



    def plotInterface(self, background = "both", annotate = True,\
                      idx = 0, verbose = 1, align_base = "cell_1", scale = False,\
                      save = False, format = "pdf", dpi = 100,\
                      col = None, row = None, N = None, handle = False):
        """Plot the vector contour of the selected interface"""

        if verbose > 0: self.printInterfaces(idx = idx)

        if not handle:
            hFig = plt.figure()
            col, row, N = (1, 1, 1)
        hAx = plt.subplot(row, col, N)

        mat = self.cell_1[idx, :, :]
        top = self.cell_2[idx, :, :]
        rot_1 = 0
        rot_2 = self.ang[idx]
        aDeg = 0

        if align_base.lower() == "cell_1":
            mat, aRad = ut.align(mat, align_to = [1, 0], verbose = verbose - 1)
            top = ut.rotate(top, aRad, verbose = verbose - 1)
            rot_1 = rot_1 + np.rad2deg(aRad)
            rot_2 = rot_2 + np.rad2deg(aRad)
        elif align_base.lower() == "both":
            mat, aRad = ut.align(mat, align_to = [1, 0], verbose = verbose - 1)
            top, bRad = ut.align(top, align_to = [1, 0], verbose = verbose - 1)
            rot_1 = rot_1 + np.rad2deg(aRad)
            rot_2 = rot_2 + np.rad2deg(bRad)
        elif align_base.lower() == "center":
            mat, aRad = ut.align(mat, align_to = [1, 0], verbose = verbose - 1)
            top, bRad = ut.center(mat, top, verbose = verbose - 1)
            rot_1 = rot_1 + np.rad2deg(aRad)
            rot_2 = rot_2 + np.rad2deg(bRad)

        if background.lower() == "cell_1" or background.lower() == "both":
            ut.overlayLattice(lat = self.base_1, latRep = self.rep_1[idx, :, :],\
                              rot = rot_1, hAx = hAx, ls = '-')

        if background.lower() == "cell_2" or background.lower() == "both":
            ut.overlayLattice(lat = self.base_2, latRep = self.rep_2[idx, :, :],\
                              rot = rot_2, hAx = hAx, ls = '--')


        """Origo to a"""
        vec_ax = np.array([0, mat[0, 0]])
        vec_ay = np.array([0, mat[1, 0]])
        hAx.plot(vec_ax, vec_ay, linewidth = 1, color = 'b', label = r"$\vec a_1$")

        """Origo to b"""
        vec_bx = np.array([0, mat[0, 1]])
        vec_by = np.array([0, mat[1, 1]])
        hAx.plot(vec_bx, vec_by, linewidth = 1, color = 'r', label = r"$\vec a_2$")

        """a to a + b"""
        vec_abx = np.array([vec_ax[1], vec_ax[1] + vec_bx[1]])
        vec_aby = np.array([vec_ay[1], vec_ay[1] + vec_by[1]])
        hAx.plot(vec_abx, vec_aby, linewidth = 1, color = 'k', ls = '-')

        """b to b + a"""
        vec_bax = np.array([vec_bx[1], vec_bx[1] + vec_ax[1]])
        vec_bay = np.array([vec_by[1], vec_by[1] + vec_ay[1]])
        hAx.plot(vec_bax, vec_bay, linewidth = 1, color = 'k', ls = '-')

        """Get the extent of the plotted figure"""
        box = np.zeros((2, 4))
        box[0, :] = np.array([np.min([vec_ax, vec_bx, vec_abx, vec_bax]),\
                              np.max([vec_ax, vec_bx, vec_abx, vec_bax]),\
                              np.min([vec_ay, vec_by, vec_aby, vec_bay]),\
                              np.max([vec_ay, vec_by, vec_aby, vec_bay])])

        """Annotate with original B lattice vectors, if selected"""
        if annotate:
            vec_ax_an = np.array([0, top[0, 0]])
            vec_ay_an = np.array([0, top[1, 0]])
            hAx.plot(vec_ax_an, vec_ay_an, linewidth = 1, color = 'b',\
                                 label = r"$\vec b_1$", ls = '--')

            vec_bx_an = np.array([0, top[0, 1]])
            vec_by_an = np.array([0, top[1, 1]])
            hAx.plot(vec_bx_an, vec_by_an, linewidth = 1, color = 'r',\
                                 label = r"$\vec b_2$", ls = '--')

            vec_abx_an = np.array([vec_ax_an[1], vec_ax_an[1] + vec_bx_an[1]])
            vec_aby_an = np.array([vec_ay_an[1], vec_ay_an[1] + vec_by_an[1]])
            hAx.plot(vec_abx_an, vec_aby_an, linewidth = 1, color = 'k', ls = '--')

            vec_bax_an = np.array([vec_bx_an[1], vec_bx_an[1] + vec_ax_an[1]])
            vec_bay_an = np.array([vec_by_an[1], vec_by_an[1] + vec_ay_an[1]])
            hAx.plot(vec_bax_an, vec_bay_an, linewidth = 1, color = 'k', ls = '--')

            """Get the extent of the plotted figure"""
            box[1, :] = np.array([np.min([vec_ax_an, vec_bx_an, vec_abx_an, vec_bax_an]),\
                                  np.max([vec_ax_an, vec_bx_an, vec_abx_an, vec_bax_an]),\
                                  np.min([vec_ay_an, vec_by_an, vec_aby_an, vec_bay_an]),\
                                  np.max([vec_ay_an, vec_by_an, vec_aby_an, vec_bay_an])])

        l1 = np.min(box[:, 1] - box[:, 0])
        l2 = np.min(box[:, 3] - box[:, 2])
        margin = np.min([4, l1, l2])

        xMin = np.min(box[:, 0]) - margin
        yMin = np.min(box[:, 2]) - margin
        xMax = np.max(box[:, 1]) + margin
        yMax = np.max(box[:, 3]) + margin

        if scale:
            xMin = np.min(box) - margin
            xMax = np.max(box) + margin
            yMin = xMin
            yMax = xMax

        hAx.set_xlim(left = xMin, right = xMax)
        hAx.set_ylim(bottom = yMin, top = yMax)

        hAx.set_title("Interface %s" % idx)
        if not handle:
            hAx.set_ylabel(r"$y, (\AA)$")
            hAx.set_xlabel(r"$x, (\AA)$")

            hAx.legend(framealpha = 1)
        else:
            if np.isin(N, range(1, row*col + 1, col)):
                hAx.set_ylabel(r"$y, (\AA)$")
            if np.isin(N, range((row - 1) * col + 1, row * col + 1)):
                hAx.set_xlabel("$x, (\AA)$")


        if handle:
            return

        plt.tight_layout()
        if save:
            if save is True:
                save_fig(filename = "interface_%s.%s" % (idx, format),\
                         format = format, dpi = dpi, verbose = verbose)
            else:
                save_fig(filename = save, format = format, dpi = dpi,\
                         verbose = verbose)
            plt.close()
        else:
            plt.show()



    def plotCombinations(self, idx = None, const = None, exp = 1,\
                         mark = None, save = False, format = "pdf",\
                         dpi = 100, handle = False, eps = "eps_mas",\
                         verbose = 1, hAx = None):
        """Plots strain vs. atoms for the interfaces"""

        if idx is None: idx = np.arange(self.atoms.shape[0])

        if not handle:
            hFig = plt.figure()
            col, row, N = (1, 1, 1)
        hAx = plt.subplot(row, col, N)
        
        atoms = self.atoms

        if eps == "eps_11":
            strain = np.abs(self.eps_11)
            if verbose > 0: print("Showing absolute value of %s" % (eps))
        elif eps == "eps_22":
            strain = np.abs(self.eps_22)
            if verbose > 0: print("Showing absolute value of %s" % (eps))
        elif eps == "eps_12":
            strain = np.abs(self.eps_12)
            if verbose > 0: print("Showing absolute value of %s" % (eps))
        else:
            strain = self.eps_mas

        if verbose > 0:
            print("Items total: %i" % idx.shape[0])

        if mark is not None:
            if isinstance(mark[0], (int, np.integer)):
                mask = np.zeros(np.shape(idx), dtype = bool)
                mask[mark] = True
            else:
                mask = mark

            strain_m = strain[mask]
            atoms_m = atoms[mask]
            strain = strain[np.logical_not(mask)]
            atoms = atoms[np.logical_not(mask)]

        if const is not None:

            """Find atom/strain pairs below limit set by atoms = A * strain ** exp"""
            low = atoms < (const * strain ** exp)
            hi = np.logical_not(low)

            hAx.plot(strain[low] * 100, atoms[low], color = 'b', marker = ".",\
                     linestyle = "None", markersize = 2)

            hAx.plot(strain[hi] * 100, atoms[hi], color = 'r', marker = ".",\
                     linestyle = "None", markersize = 2)

            """Plot the dividing line for the specified limit"""
            j = np.lexsort((atoms, strain))[0]
            k = np.lexsort((strain, atoms))[0]
            x = np.logspace(np.log(strain[j] * 0.9), np.log(strain[k] * 1.1), 1000, base = np.exp(1))

            hAx.plot(x * 100, const * x ** exp, linewidth = 0.5, color = 'k')

            if verbose > 0:
                print("Items below: %i | Items above: %i" % (np.sum(low), np.sum(hi)))
        else:
            hAx.plot(strain * 100, atoms, color = 'b', marker = ".",\
                     linestyle = "None", markersize = 2)

        if mark is not None:
            hAx.plot(strain_m * 100, atoms_m, color = 'g', marker = ".",\
                     linestyle = "None", markersize = 3)
            if verbose > 0:
                print("Items marked: %i" % strain_m.shape[0])

        hAx.set_xscale("log")
        hAx.set_yscale("log")
        hAx.set_ylabel("Nr of Atoms")
        hAx.set_xlabel(r"Strain, %s, (%%)" % eps)
        hAx.set_title("Created Interfaces")

        if handle:
            return

        plt.tight_layout()
        if save:
            if save is True:
                save_fig(filename = "combinations.%s" % format, format = format,\
                         dpi = dpi, verbose = verbose)
            else:
                save_fig(filename = save, format = format, dpi = dpi,\
                         verbose = verbose)
            plt.close()
        else:
            plt.show()



    def saveInterfaces(self, filename = "Interfaces.pkl", verbose = 1):
        """Function or saveing an interface collection to a .pyz file"""

        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as wf:
            pickle.dump(self, wf, -1)

        if verbose > 0:
            print("Data saved to: %s" % filename)



    def writeToExcel(self, filename = "Interfaces.xlsx", idx = None, prec = 4,\
                     verbose = 1):
        """Function for writing specified interfaces to an excel-file"""

        if idx is None:
            idx = np.arange(self.atoms.shape[0])
        elif type(idx) is int: 
            idx = np.array([idx])
        else:
            idx = np.array(idx)
        

        dataDict = {"Index": np.arange(idx.shape[0]), "Original Rotation": self.ang[idx],\
                    "Length a": np.round(self.getBaseLengths(cell = 1)[idx, 0], prec),\
                    "Length b": np.round(self.getBaseLengths(cell = 1)[idx, 1], prec),\
                    "Angle a/b": np.round(self.getBaseAngles(cell = 1)[idx], prec),\
                    "Atoms": self.atoms[idx],\
                    "Area": self.getAreas()[idx],\
                    "Strain 11": np.round(self.eps_11[idx], prec),\
                    "Strain 22": np.round(self.eps_22[idx], prec),\
                    "Strain 12": np.round(self.eps_12[idx], prec),\
                    "Strain MAS": np.round(self.eps_mas[idx], prec),\
                    "E_int": self.e_int[idx],\
                    "Base 1 ax": np.round(self.cell_1[idx, 0, 0], prec),\
                    "Base 1 ay": np.round(self.cell_1[idx, 1, 0], prec),\
                    "Base 1 bx": np.round(self.cell_1[idx, 0, 1], prec),\
                    "Base 1 by": np.round(self.cell_1[idx, 1, 1], prec),\
                    "Base 2 ax": np.round(self.cell_2[idx, 0, 0], prec),\
                    "Base 2 ay": np.round(self.cell_2[idx, 1, 0], prec),\
                    "Base 2 bx": np.round(self.cell_2[idx, 0, 1], prec),\
                    "Base 2 by": np.round(self.cell_2[idx, 1, 1], prec),\
                    "Rep 1 ax": np.round(self.rep_1[idx, 0, 0], prec),\
                    "Rep 1 ay": np.round(self.rep_1[idx, 1, 0], prec),\
                    "Rep 1 bx": np.round(self.rep_1[idx, 0, 1], prec),\
                    "Rep 1 by": np.round(self.rep_1[idx, 1, 1], prec),\
                    "Rep 2 ax": np.round(self.rep_2[idx, 0, 0], prec),\
                    "Rep 2 ay": np.round(self.rep_2[idx, 1, 0], prec),\
                    "Rep 2 bx": np.round(self.rep_2[idx, 0, 1], prec),\
                    "Rep 2 by": np.round(self.rep_2[idx, 1, 1], prec)}

        data = pd.DataFrame(dataDict)
        data.to_excel(filename)

        if verbose > 0:
            string = "Data written to Excel file: %s" % filename
            print("\n" + "=" * len(string))
            print("%s" % string)
            print("=" * len(string) + "\n")





    def buildInterface(self, idx = 0, z_1 = 1, z_2 = 1, d = 2.5,\
                       verbose = 1, vacuum = 0, translate = None,\
                       surface = None):
        """Function for build a specific interface"""

        if verbose > 0:
            self.printInterfaces(idx = idx)

        """The strained new basis"""
        F = np.zeros((3, 3))
        F[2, 2] = self.base_1[2, 2] * z_1 + self.base_2[2, 2] * z_2 + d
        F[0:2, 0:2] = self.cell_1[idx, :, :]

        """The unstrained new basis"""
        D = np.zeros((3, 3))
        D[2, 2] = self.base_2[2, 2] * z_2
        D[0:2, 0:2] = self.cell_2[idx, :, :]

        """Working on interface A"""
        """Set up the bottom interface with the correct repetitions"""
        rep_1 = np.zeros((3, 3))
        rep_1[0:2, 0:2] = self.rep_1[idx, :, :]

        """Set all hi-lo limits for the cell repetitions"""
        h = 10
        rep_1 = [rep_1[0, :].min() - h, rep_1[0, :].max() + h,\
                 rep_1[1, :].min() - h, rep_1[1, :].max() + h,\
                 0, z_1 - 1]

        """Extend the cell as spcefied"""
        pos_1_ext, spec_1 = ut.extendCell(base = self.base_1, rep = rep_1,\
                                          pos = self.pos_1.T, spec = self.spec_1)

        """Working on interface B"""    
        """Set up the top interface with the correct repetitions and rotation"""
        rep_2 = np.zeros((3, 3))
        rep_2[0:2, 0:2] = self.rep_2[idx, :, :]

        """Set all hi-lo limits for the cell repetitions"""
        h = 10
        rep_2 = [rep_2[0, :].min() - h, rep_2[0, :].max() + h,\
                 rep_2[1, :].min() - h, rep_2[1, :].max() + h,\
                 0, z_2 - 1]

        """Extend the cell as specified"""
        pos_2_ext, spec_2 = ut.extendCell(base = self.base_2, rep = rep_2,\
                                          pos = self.pos_2.T, spec = self.spec_2)

        """Initial rotation"""
        initRot = np.deg2rad(self.ang[idx])

        """Rotate the positions pos_rot = R*pos"""
        pos_2_ext_rot = ut.rotate(pos_2_ext, initRot, verbose = verbose - 1)

        """Convert to direct coordinates in the unstrained D base"""
        pos_2_d_D = np.matmul(np.linalg.inv(D), pos_2_ext_rot)

        """Convert the cell back to Cartesian using the strained basis.
        But with the Z parameter as in the D cell"""
        temp_F = F.copy()
        temp_F[2, 2] = D[2, 2]
        pos_2_F = np.matmul(temp_F, pos_2_d_D)

        """Combine the positions of the two cells"""
        pos = np.zeros((3, pos_1_ext.shape[1] + pos_2_F.shape[1]))
        pos[:, :pos_1_ext.shape[1]] = pos_1_ext

        """Shift Z positions of top cell to (cell_A + d)"""
        pos_2_F[2, :] = pos_2_F[2, :] + self.base_1[2, 2] * z_1 + d
        pos[:, pos_1_ext.shape[1]:] = pos_2_F

        """If a translation is specified shift (x,y) coordinates of top cell accordingly""" 
        if translate is not None:
            T = ut.getTranslation(translate, surface, verbose = verbose)
            cT = np.matmul(self.base_1, T)
            pos[:, pos_1_ext.shape[1]:] = pos[:, pos_1_ext.shape[1]:] + cT[:, np.newaxis]
            if verbose: 
                string = "Translation [%.2f, %.2f, %.2f] (C) or [%.2f, %.2f, %.2f] (D)"\
                         % (cT[0], cT[1], cT[2], T[0], T[1], T[2])
                ut.infoPrint(string)

        """Convert the entire new cell to direct coordinates, add d above as well"""
        F[2, 2] += d
        pos_d = np.matmul(np.linalg.inv(F), pos)

        """Remove all positions outside [0, 1)"""
        pos_d = np.round(pos_d, 8)
        F = np.round(F, 8)

        keep = np.all(pos_d < 1, axis = 0) * np.all(pos_d >= 0, axis = 0)
        pos_d = pos_d[:, keep]
        species = np.concatenate((spec_1, spec_2))[keep]

        """Return to cartesian coordinates and change shape to (...,3)"""
        pos = np.matmul(F, pos_d).T

        """Add vacuum if specified"""
        F[2, 2] = F[2, 2] + vacuum
        if verbose: 
            string = "Distance added (between,above): %.2f | Vacuum added (above): %.2f"\
                     % (d, vacuum)
            ut.infoPrint(string)

        return F, pos, species



    def exportInterface(self, idx = 0, z_1 = 1, z_2 = 1, d = 2.5,\
                        verbose = 1, mass = None, format = "lammps",\
                        filename = None, vacuum = 0, translate = None,\
                        surface = None):
        """Function for writing an interface to a specific file format"""

        if filename is None:
            if translate is None:
                filename = "interface_%s.%s" % (idx, format)
            else:
                filename = "interface_%s_T%s.%s" % (idx, translate, format)

        """Build the selected interface"""
        base, pos, type_n = self.buildInterface(idx = idx, z_1 = z_1, z_2 = z_2, d = d,\
                                                verbose = verbose, vacuum = vacuum,\
                                                translate = translate, surface = surface)

        """Sort first based on type then Z-position then Y-position"""
        ls = np.lexsort((pos[:, 1], pos[:, 2], type_n))
        type_n = type_n[ls]
        pos = pos[ls]

        """After sorting, index all positions"""
        index = np.arange(type_n.shape[0])

        """Build an Atoms object"""
        atoms = structure.Structure(cell = base, pos = pos, type_n = type_n, type_i = None,\
                                    mass = mass, idx = index, filename = filename, pos_type = "c")

        """Align the first dimension to the x-axis"""
        atoms.alignStructure(dim = [1, 0, 0], align = [1, 0, 0])

        """Write the structure object to specified file"""
        atoms.writeStructure(filename = filename, format = format, verbose = verbose)



    def buildSurface(self, idx = 0, surface = 1, z = 1, verbose = 1, vacuum = 0):
        """Function for build a specific interface"""
        
        if verbose > 0:
            self.printInterfaces(idx = idx)
            string = "Building surface nr %i in the interface shown above" % surface
            ut.infoPrint(string)

        """Basis for the selected surface, and the cell repetitions"""
        rep = np.zeros((3, 3))
        new_base = np.zeros((3, 3))
        if surface == 1:
            old_base = self.base_1
            spec = self.spec_1
            pos = self.pos_1

            new_base[2, 2] = self.base_1[2, 2] * z
            new_base[0:2, 0:2] = self.cell_1[idx, :, :]
            
            rep[0:2, 0:2] = self.rep_1[idx, :, :]
        elif surface == 2:
            old_base = self.base_2
            spec = self.spec_2
            pos = self.pos_2

            new_base[2, 2] = self.base_2[2, 2] * z
            new_base[0:2, 0:2] = self.cell_2[idx, :, :]

            rep[0:2, 0:2] = self.rep_2[idx, :, :]

        """Set all hi-lo limits for the cell repetitions"""
        h = 10
        rep = [rep[0, :].min() - h, rep[0, :].max() + h,\
               rep[1, :].min() - h, rep[1, :].max() + h,\
               0, z - 1]

        """Extend the cell as spcefied"""
        pos_ext, spec_ext = ut.extendCell(base = old_base, rep = rep,\
                                          pos = pos.T, spec = spec)

        """Convert the entire new cell to direct coordinates"""
        pos_d = np.matmul(np.linalg.inv(new_base), pos_ext)

        """Remove all positions outside [0, 1)"""
        pos_d = np.round(pos_d, 8)
        new_base = np.round(new_base, 8)

        keep = np.all(pos_d < 1, axis = 0) * np.all(pos_d >= 0, axis = 0)
        pos_d = pos_d[:, keep]
        species = spec_ext[keep]

        """Return to cartesian coordinates and change shape to (...,3)"""
        pos = np.matmul(new_base, pos_d).T

        """Add vacuum if specified"""
        new_base[2, 2] = new_base[2, 2] + vacuum
        if verbose: 
            string = "Vacuum added: %.2f"\
                     % (vacuum)
            ut.infoPrint(string)

        return new_base, pos, species



    def exportSurface(self, idx = 0, z = 1, verbose = 1, mass = None, format = "lammps",\
                        filename = None, vacuum = 0, surface = 1):
        """Function for writing an interface to a specific file format"""

        if filename is None:
            filename = "surface_%s_S%s.%s" % (idx, surface, format)

        """Build the selected interface"""
        base, pos, type_n = self.buildSurface(idx = idx, z = z, verbose = verbose,\
                                              vacuum = vacuum, surface = surface)

        """Sort first based on type then Z-position then Y-position"""
        ls = np.lexsort((pos[:, 1], pos[:, 2], type_n))
        type_n = type_n[ls]
        pos = pos[ls]

        """After sorting, index all positions"""
        index = np.arange(type_n.shape[0])

        """Build an Atoms object"""
        atoms = structure.Structure(cell = base, pos = pos, type_n = type_n, type_i = None,\
                                    mass = mass, idx = index, filename = filename, pos_type = "c")

        """Align the first dimension to the x-axis"""
        atoms.alignStructure(dim = [1, 0, 0], align = [1, 0, 0])

        """Write the structure object to specified file"""
        atoms.writeStructure(filename = filename, format = format, verbose = verbose)




    def buildInterfaceStructure(self, idx = 0, z_1 = 1, z_2 = 1, d = 2.5,\
                                verbose = 1, mass = None, filename = None,\
                                vacuum = 0, translate = None, surface = None):
        """Function for writing an interface to a specific file format"""

        if filename is None:
            if translate is None:
                filename = "InterfaceStructure_%s" % (idx)
            else:
                filename = "InterfaceStructure_%s_T%s" % (idx, translate)

        """Build the selected interface"""
        base, pos, type_n = self.buildInterface(idx = idx, z_1 = z_1, z_2 = z_2, d = d,\
                                                verbose = verbose, vacuum = vacuum,\
                                                translate = translate, surface = surface)

        """Sort first based on type then Z-position then Y-position"""
        ls = np.lexsort((pos[:, 1], pos[:, 2], type_n))
        type_n = type_n[ls]
        pos = pos[ls]

        """After sorting, index all positions"""
        index = np.arange(type_n.shape[0])

        """Build an Atoms object"""
        atoms = structure.Structure(cell = base, pos = pos, type_n = type_n, type_i = None,\
                                    mass = mass, idx = index, filename = filename, pos_type = "c")

        """Align the first dimension to the x-axis"""
        atoms.alignStructure(dim = [1, 0, 0], align = [1, 0, 0])

        """Retrun the structure object"""
        return atoms



    def buildSurfaceStructure(self, idx = 0, z = 1, verbose = 1, mass = None,\
                              filename = None, vacuum = 0, surface = None):
        """Function for writing an interface to a specific file format"""

        if filename is None:
            filename = "SurfaceStructure_%s_S%s" % (idx, surface)

        """Build the selected interface"""
        base, pos, type_n = self.buildSurface(idx = idx, z = z, verbose = verbose,\
                                              vacuum = vacuum, surface = surface)

        """Sort first based on type then Z-position then Y-position"""
        ls = np.lexsort((pos[:, 1], pos[:, 2], type_n))
        type_n = type_n[ls]
        pos = pos[ls]

        """After sorting, index all positions"""
        index = np.arange(type_n.shape[0])

        """Build an Atoms object"""
        atoms = structure.Structure(cell = base, pos = pos, type_n = type_n, type_i = None,\
                                    mass = mass, idx = index, filename = filename, pos_type = "c")

        """Align the first dimension to the x-axis"""
        atoms.alignStructure(dim = [1, 0, 0], align = [1, 0, 0])

        """Retrun the structure object"""
        return atoms
