#!/usr/bin/env python3

import pickle
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from interfaceBuilder import structure
from interfaceBuilder import utils as ut
from interfaceBuilder import file_io

class Interface():
    """
    Class for building and holdiding a collection of generated interfaces.
    Including methods for analyzing them.
    """

    def __init__(self, structure_a, structure_b):
        """Constructor, build an interface object using 2 structure objects as input

        structure_a = structure.Structure(), Bottom surface
        
        structure_b = structure.Structure(), Top surface
        """

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
        self.translations = None

        """E_s1/A + E_s2/A - E_i/A(slabs (unstrained reference) - Interface)"""
        self.w_sep_c = None
        self.w_sep_d = None

        """E_s1/A + E_s2/A - E_i/A (slabs (strained reference) - Interface)"""
        self.w_seps_c = None
        self.w_seps_d = None

        """E_i/A - E_b1/A - E_b2/A (Interface - bulk)"""
        self.e_int_c = None
        self.e_int_d = None

        self.base_1 = structure_a.cell
        self.base_2 = structure_b.cell
        self.pos_1 = structure_a.pos
        self.pos_2 = structure_b.pos
        self.spec_1 = structure_a.type_n
        self.spec_2 = structure_b.type_n
        self.mass_1 = structure_a.mass
        self.mass_2 = structure_b.mass
        self.filename = None
        self.alt_base = []



    def deleteInterfaces(self, keep, verbose = 1):
        """Function for removing interfaces.

        keep = array(bool/index), Boolean mask or index of interfaces 
        to keep

        verbose = int, Print extra information
        """

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
        self.e_int_c = self.e_int_c[keep]
        self.e_int_d = self.e_int_d[keep]
        self.w_sep_c = self.w_sep_c[keep]
        self.w_sep_d = self.w_sep_d[keep]
        self.w_seps_c = self.w_seps_c[keep]
        self.w_seps_d = self.w_seps_d[keep]

        if verbose > 0:
            string = "Interfaces deleted: %i | Interfaces remaining: %i"\
                     % (np.sum(np.logical_not(keep)), np.sum(keep))
            ut.infoPrint(string)


    def setFilename(self, filename, verbose = 1):
        """Function to set filename

        filename = str, Name to set

        verbose = int, Print extra information
        """
        
        self.filename = filename
        if verbose > 0:
            string = "Filename updated to: %s" % filename
            ut.infoPrint(string)


    def sortInterfaces(self, sort = "atoms", opt = None, rev = False):
        """Function for sorting the interfaces bases on predefined properties

        Options to sort by
        ------------------
        atoms = Nr of atoms
        angle = a/b cell angles
        area = area of the interfaces
        eps_11 = eps_11
        eps_22 = eps_22
        eps_12 = eps_12
        eps_mas = eps_mas
        base_angle_1 = base angle of bottom cell
        base_angle_2 = base angle of top cell
        w_sep_c = work of sepparation, unstrained reference
        w_seps_c = work of sepparation, strained reference
        w_sep_d = DFT level work of sepparation, unstrained reference
        w_seps_d = DFT level work of sepparation, strained reference
        e_int_c = interfacial energy
        e_int_d = DFT level interfacial energy

        rev = True/False, Reverse sort order

        opt = int, Index if sorting by an array property, i.e. w_sep*/e_int*
        """

        sort = sort.lower()
        sortable_properties = ["atoms",   "angle",  "area", "e_int_d",\
                               "eps_11",  "eps_22", "eps_12", "w_sep_d",\
                               "eps_mas", "e_int_c", "w_sep_c", "w_seps_c",\
                               "base_angle_1", "base_angle_2", "w_seps_d"]

        if sort == "atoms":
            si = np.argsort(self.atoms)
        elif sort == "angle":
            si = np.argsort(self.ang)
        elif sort == "e_int_c":
            """Opt int corresponding to chosen translation (0-based as translation)"""
            si = np.argsort(self.e_int_c[opt])
        elif sort == "w_sep_c":
            """Opt int corresponding to chosen translation (0-based as translation)"""
            si = np.argsort(self.w_sep_c[opt])
        elif sort == "w_seps_c":
            """Opt int corresponding to chosen translation (0-based as translation)"""
            si = np.argsort(self.w_seps_c[opt])
        elif sort == "e_int_d":
            """Opt int corresponding to chosen translation (0-based as translation)"""
            si = np.argsort(self.e_int_d[opt])
        elif sort == "w_sep_d":
            """Opt int corresponding to chosen translation (0-based as translation)"""
            si = np.argsort(self.w_sep_d[opt])
        elif sort == "w_seps_d":
            """Opt int corresponding to chosen translation (0-based as translation)"""
            si = np.argsort(self.w_seps_d[opt])
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
        """Sort interfaces based on supplied index

        index = array([N,]), Sort index to sort the interfaces by
        """

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
        self.e_int_c = self.e_int_c[index]
        self.w_sep_c = self.w_sep_c[index]
        self.w_seps_c = self.w_seps_c[index]

        self.e_int_d = self.e_int_d[index]
        self.w_sep_d = self.w_sep_d[index]
        self.w_seps_d = self.w_seps_d[index]


    def getAtomStrainDuplicates(self, tol_mag = 7, verbose = 1):
        """Get the index of interfaces with strains within specified tolerences

        tol_mag = int, Magnitude at which to consider stains identical
        7 mean rounded to 7 decimals

        verbose = int, Print extra information
        """

        """Find unique strains within specified tolerances, favor lowest number of atoms"""
        values = np.zeros((self.atoms.shape[0], 2))
        values[:, 0] = self.atoms.copy()
        values[:, 1] = np.round(self.eps_mas.copy(), tol_mag)
        unique = np.unique(values, axis = 0, return_index = True)[1]
        index = np.in1d(np.arange(self.atoms.shape[0]), unique)

        if verbose > 0:
            string = "Unique strain/atom combinations found: %i within a tolerance magnitude of 1e-%i"\
                     % (np.sum(index), tol_mag)
            ut.infoPrint(string)

        return index



    def removeAtomStrainDuplicates(self, tol_mag = 7, verbose = 1):
        """Remove interfaces based on duplicates of atom/strain

        tol_mag = int, Magnitude at which to consider stains identical
        7 mean rounded to 7 decimals

        verbose = int, Print extra information
        """

        keep = self.getAtomStrainDuplicates(tol_mag = tol_mag, verbose = verbose - 1)

        self.deleteInterfaces(keep = keep, verbose = verbose)



    def changeBaseElements(self, change = None, swap = None,\
                           cell = 1, verbose = 1):
        """Function for changing the composition of the base lattice
           
        change - Dict with keys {"from": "XX", "to": "YY", "mass": MASS}
        changes the spec = XX to spec = YY and changes mass to MASS

        swap - Dict with keys {"swap_1: "XX", "swap_2": "YY"}
        switches the spec = XX and spec = YY and switches mass as well

        cell = int, Bottom (1) or top (2) cell

        verbose = int, Print extra information
        """

        if (change is not None) and (swap is not None):
            string = "Cant use both change and swap at the same time"
            ut.infoPrint(string)

        elif change is not None:
            if type(change["from"]) == str: change["from"] = bytes(change["from"], "utf-8")
            if type(change["to"]) == str: change["to"] = bytes(change["to"], "utf-8")

            if cell == 1:
                self.mass_1[self.spec_1 == change["from"]] = change["mass"]
                self.spec_1[self.spec_1 == change["from"]] = change["to"]
            elif cell == 2:
                self.mass_2[self.spec_2 == change["from"]] = change["mass"]
                self.spec_2[self.spec_2 == change["from"]] = change["to"]
            else:
                return
            
            if verbose > 0:
                string = "Changing elements: %s --> %s and updating mass to: %.4f for cell %i"\
                         % (change["from"].decode("utf-8"), change["to"].decode("utf-8"),\
                            change["mass"], cell)
                ut.infoPrint(string)

        elif swap is not None:
            if type(swap["swap_1"]) == str: swap["swap_1"] = bytes(swap["swap_1"], "utf-8")
            if type(swap["swap_2"]) == str: swap["swap_2"] = bytes(swap["swap_2"], "utf-8")

            if cell == 1:
                mass1 = self.mass_1[self.spec_1 == swap["swap_1"]][0]
                spec1 = self.spec_1[self.spec_1 == swap["swap_1"]][0]
                mask1 = self.spec_1 == swap["swap_1"]

                mass2 = self.mass_1[self.spec_1 == swap["swap_2"]][0]
                spec2 = self.spec_1[self.spec_1 == swap["swap_2"]][0]
                mask2 = self.spec_1 == swap["swap_2"]

                self.mass_1[mask1] = mass2
                self.spec_1[mask1] = spec2
                self.mass_1[mask2] = mass1
                self.spec_1[mask2] = spec1

            elif cell == 2:
                mass1 = self.mass_2[self.spec_2 == swap["swap_1"]][0]
                spec1 = self.spec_2[self.spec_2 == swap["swap_1"]][0]
                mask1 = self.spec_2 == swap["swap_1"]

                mass2 = self.mass_2[self.spec_2 == swap["swap_2"]][0]
                spec2 = self.spec_2[self.spec_2 == swap["swap_2"]][0]
                mask2 = self.spec_2 == swap["swap_2"]

                self.mass_2[mask1] = mass2
                self.spec_2[mask1] = spec2
                self.mass_2[mask2] = mass1
                self.spec_2[mask2] = spec1

            else:
                return
            
            if verbose > 0:
                string = "Swaping elements: %s and %s and swaping masses: %.4f to %.4f for cell %i"\
                         % (swap["swap_1"].decode("utf-8"), swap["swap_2"].decode("utf-8"),\
                            mass1, mass2, cell)
                ut.infoPrint(string)

        else:
            return


    def getStrain(self, idx = None, strain = "eps_mas", base_1 = None, base_2 = None):
        """Get specified strain with regards to the specified base

        idx = array([int,]), Index of interfaces to consider
        
        strain = str("eps_mas"/"eps_11"/"eps_22"/"eps_12"/"array"), strain to return
        array returns (3,N) array with [[eps_11],[eps_22],[eps_12]]

        base_1 = array([2,2]), Base to use for the bottom structure.
        None uses standard base

        base_1 = array([2,2]), Base to use for the top structure
        None uses standard base
        """

        """Check if strain keyword is supported"""
        strain_avail = ["eps_11", "eps_22", "eps_12", "eps_mas", "array"]
        if strain.lower() not in strain_avail:
            string = "Unrecognized strain argument: %s" % strain
            ut.infoPrint(string)
            return

        """Set some defaults"""
        if idx is None: idx = np.arange(self.atoms.shape[0])
        if isinstance(idx, (int, np.integer)): idx = [idx]

        if base_1 is None and base_2 is None:
            """Send back the original strains as specified"""
            if strain.lower() == "eps_11":
                return self.eps_11[idx]
            elif strain.lower() == "eps_22":
                return self.eps_22[idx]
            elif strain.lower() == "eps_12":
                return self.eps_12[idx]
            elif strain.lower() == "eps_mas":
                return self.eps_mas[idx]
            elif strain.lower() == "array":
                """Return (3,N) array with [[eps_11],[eps_22],[eps_12]]"""
                return np.vstack((self.eps_11[idx],\
                                  self.eps_22[idx],\
                                  self.eps_12[idx]))

        else:
            """Calculate new cell vectors baded on the supplied base and the 
            existing repetitions of that base"""

            """Change the bottom cell vectors to match the new base"""
            if base_1 is not None:
                if isinstance(base_1, (int, np.integer)): 
                    base_1 = self.alt_base[base_1]
                A = np.matmul(base_1[:2, :2], self.rep_1[idx, :, :])
            else:
                A = self.cell_1[idx, :, :]
            if base_2 is not None:
                if isinstance(base_2, (int, np.integer)): 
                    base_2 = self.alt_base[base_2]
                B = np.matmul(base_2[:2, :2], self.rep_2[idx, :, :])
            else:
                B = self.cell_2[idx, :, :]

            """Get the new strains"""
            eps_11, eps_22, eps_12, eps_mas = ut.calcStrains(a = A, b = B)

            if strain.lower() == "eps_11":
                return eps_11
            elif strain.lower() == "eps_22":
                return eps_22
            elif strain.lower() == "eps_12":
                return eps_12
            elif strain.lower() == "eps_mas":
                return eps_mas
            elif strain.lower() == "array":
                """Return (3,N) array with [[eps_11],[eps_22],[eps_12]]"""
                return np.vstack((eps_11, eps_22, eps_12))
        

    def getAtomStrainRatio(self, strain = "eps_mas", const = None, exp = 1, verbose = 1,\
                           base_1 = None, base_2 = None):
        """Get the property atoms - A * abs(strain) ** B

        strain = str("eps_11"/"eps_22"/"eps_12"/"eps_mas"), Streain component 
        to use in the calculation
        
        const = float, Use this specific value for the A parameter, 
        if None then the deafult values for A and B are used

        exp = float, use this specific value for the B parameter

        base_1 = array([2,2]), Calculate the strains using this base for
        the bottom cell. None uses standard base
        
        base_2 = array([2,2]), Calculate the strains using this base for
        the top cell. None uses standard base

        verbose = int, Print extra information
        """

        if const is None:
            const, exp = self.getAtomStrainExpression(strain = strain, verbose = verbose - 1,\
                                                      base_1 = base_1, base_2 = base_2)

        if verbose > 0: 
            string = "Returning values of expression: atoms - A * |strain|^B with A,B: %.3e, %.3e"\
                     % (const, exp)
            ut.infoPrint(string)
    
        eps = self.getStrain(idx = None, strain = strain, base_1 = base_1, base_2 = base_2)
        return self.atoms - const * np.abs(eps) ** exp


    def getAtomStrainMatches(self, matches = 100, const = None, exp = 1,\
                             strain = "eps_mas", verbose = 1, max_iter = 500,\
                             tol = 1e-7, endpoint = "over", base_1 = None,\
                             base_2 = None):
        """Function for returning interfaces matching the critera
        atoms - A * abs(strain) ** B
    
        strain = str("eps_11"/"eps_22"/"eps_12"/"eps_mas"), Streain component 
        to use in the calculation
        
        const = float, Use this specific value for the A parameter, 
        if None then the deafult values for A and B are used

        exp = float, use this specific value for the B parameter

        base_1 = array([2,2]), Calculate the strains using this base for
        the bottom cell. None uses standard base
        
        base_2 = array([2,2]), Calculate the strains using this base for
        the top cell. None uses standard base

        verbose = int, Print extra information

        max_iter = int, Max number of recursiv iterations
    
        tol = float, Tolerance for aborting if other criteria is not met

        endpoint = str("over"/"under"), If aborted due to tolerance match
        then this determines if the results will include as close to 
        but below the specified matches or above the specified matches
        """

        if const is None:
            const, exp = self.getAtomStrainExpression(strain = strain, verbose = verbose - 1,\
                                                      base_1 = base_1, base_2 = base_2)

        eps = self.getStrain(idx = None, strain = strain, base_1 = base_1, base_2 = base_2)

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


    def getAtomStrainIdx(self, matches = 100, const = None, exp = 1,\
                             strain = "eps_mas", verbose = 1, max_iter = 500,\
                             tol = 1e-7, endpoint = "over", base_1 = None, base_2 = None):
        """Function for returning the index of itnterfaces below the specified ratio"""

        C, E = self.getAtomStrainMatches(matches = matches, const = const, exp = exp,\
                                         strain = strain, verbose = verbose - 1,\
                                         max_iter = max_iter, tol = tol, endpoint = endpoint,\
                                         base_1 = base_1, base_2 = base_2)

        r = self.getAtomStrainRatio(strain = strain, const = C, exp = E, verbose = verbose - 1,\
                                    base_1 = base_1, base_2 = base_2)
        idx = np.arange(self.atoms.shape[0])[r < 0]
    
        return idx



    def removeByAtomStrain(self, keep = 50000, verbose = 1, endpoint = "over",\
                           strain = "eps_mas", tol = 1e-7, max_iter = 350,\
                           base_1 = None, base_2 = None):
        """Function for removing interfaces based on Atom strain ratios"""
        
        if self.atoms.shape[0] < keep:
            return

        C, E = self.getAtomStrainMatches(matches = keep, strain = strain, verbose = verbose,\
                                    max_iter = max_iter, tol = tol, endpoint = endpoint,\
                                    base_1 = base_1, base_2 = base_2)

        r = self.getAtomStrainRatio(const = C, exp = E, strain = strain, verbose = verbose,\
                                    base_1 = base_1, base_2 = base_2)
        self.deleteInterfaces(keep = (r < 0), verbose = verbose)
        
        self.indexSortInterfaces(index = np.argsort(r[r < 0]))
        if verbose > 0:
            string = "Sorted by Atom Strain ratio atom - A*abs(strain)**B with A,B: %.3e, %.3e" % (C, E)
            ut.infoPrint(string)



    def getAtomStrainExpression(self, strain = "eps_mas", verbose = 1, base_1 = None, base_2 = None):
        """Get the curve from min(log(abs(strain))) --> min(log(atoms))"""

        """Ignore matches which are exactly 0 in the construction of the expression"""
        tol = 1e-12

        eps_sel = self.getStrain(idx = None, strain = strain, base_1 = base_1, base_2 = base_2)

        si1 = np.lexsort((self.atoms * -1, np.abs(eps_sel)))
        si2 = np.lexsort((np.abs(eps_sel), self.atoms))
            
        eps_idx = np.argmax(np.abs(eps_sel[si1][0] - eps_sel[si2]) > tol)
        eps = np.array([np.abs(eps_sel[si1][0]), np.abs(eps_sel[si2][eps_idx])])
        atoms = np.array([self.atoms[si1][0], self.atoms[si2][eps_idx]])

        eps = np.log(eps)
        atoms = np.log(atoms)

        """In y = A*x**B find A and B"""
        B = (atoms[1] - atoms[0]) / (eps[1] - eps[0])
        A = np.exp(atoms[0] - B * eps[0])
        
        if verbose > 0:
            string = "Expression found (A * x^B): %.3e * x^%.3e" % (A, B)
            ut.infoPrint(string)

        return A, B


    
    def setArrayProperty(self, prop, idx, t, value, verbose = 1):
        """Function for seting, i.e. loading back, values for w_sep* and e_int*
        parameters calculated elsewere (LAMMPS,VASP).

        prop = str("w_sep_c"/"w_seps_c"/"w_sep_d"/"w_seps_d"/
                   "e_int_c"/e_int_d), Property to set.

        idx = int or array([N,]), Index of interface to set value for

        t = int or array([N,]), Index of translation to set

        value = float or array([index, translation]), Value of property to set

        verbose = int, Print extra information
        """

        existing_props = ["w_sep_c", "w_sep_d", "w_seps_c",\
                          "w_seps_d", "e_int_c", "e_int_d"]

        if prop.lower() not in existing_props:
            string = "Unknown property: %s" % prop
            ut.infoPrint(string)
            return

        """Check some defaults"""
        if isinstance(idx, (int, np.integer)): idx = [idx]
        if isinstance(t, (int, np.integer)): t = [t]
        if np.ndim(value) == 0:
            value = np.array(value)[None, None]
        elif np.ndim(value) == 1:
            if len(idx) > 1 and len(t) == 1:
                value = np.array(value)[:, None]
            elif len(t) > 1 and len(idx) == 1:
                value = np.array(value)[None, :]
            else:
                string = "Unless idx or translation has ndim=0/1 value must be [idx,translation]"
                ut.infoPrint(string)
                return
        elif np.ndim(value) == 2:
            value = np.array(value)
            if np.shape(value) != (len(idx), len(t)):
                string = "Dim of [idx, trans] (%i,%i) != dim of value (%i,%i)"\
                         % (len(idx), len(t), value.shape[0], value.shape[1])
                ut.infoPrint(string)
                return

        mt = np.max(t)
        if prop.lower() == "w_sep_c":
            s = self.w_sep_c.shape

            if (mt + 1) > s[1]:
                self.w_sep_c = np.concatenate((self.w_sep_c,\
                                             np.zeros((s[0], (mt + 1) - s[1]))), axis = 1)

            for i, item in enumerate(t):
                self.w_sep_c[idx, item] = value[:, i]

        elif prop.lower() == "w_seps_c":
            s = self.w_seps_c.shape

            if (mt + 1) > s[1]:
                self.w_seps_c = np.concatenate((self.w_seps_c,\
                                             np.zeros((s[0], (mt + 1) - s[1]))), axis = 1)

            for i, item in enumerate(t):
                self.w_seps_c[idx, item] = value[:, i]

        elif prop.lower() == "w_sep_d":
            s = self.w_sep_d.shape

            if (mt + 1) > s[1]:
                self.w_sep_d = np.concatenate((self.w_sep_d,\
                                             np.zeros((s[0], (mt + 1) - s[1]))), axis = 1)

            for i, item in enumerate(t):
                self.w_sep_d[idx, item] = value[:, i]

        elif prop.lower() == "w_seps_d":
            s = self.w_seps_d.shape

            if (mt + 1) > s[1]:
                self.w_seps_d = np.concatenate((self.w_seps_d,\
                                             np.zeros((s[0], (mt + 1) - s[1]))), axis = 1)

            for i, item in enumerate(t):
                self.w_seps_d[idx, item] = value[:, i]

        elif prop.lower() == "e_int_c":
            s = self.e_int_c.shape

            if (mt + 1) > s[1]:
                self.e_int_c = np.concatenate((self.e_int_c,\
                                             np.zeros((s[0], (mt + 1) - s[1]))), axis = 1)

            for i, item in enumerate(t):
                self.e_int_c[idx, item] = value[:, i]

        elif prop.lower() == "e_int_d":
            s = self.e_int_d.shape

            if (mt + 1) > s[1]:
                self.e_int_d = np.concatenate((self.e_int_d,\
                                             np.zeros((s[0], (mt + 1) - s[1]))), axis = 1)

            for i, item in enumerate(t):
                self.e_int_d[idx, item] = value[:, i]

        if verbose > 0 and (mt + 1) > s[1]:
            string = "Extending %s from shape (%i,%i) to (%i,%i)"\
                     % (prop, s[0], s[1], s[0], (mt + 1))
            ut.infoPrint(string)
        if verbose > 0 and (len(idx) * len(t)) > 1:
            string = "%ix%i values for property %s have been set"\
                     % (np.shape(value)[0], np.shape(value)[1], prop)
            ut.infoPrint(string)
        elif verbose > 0:
            string = "Value for %s set to: %.3f at idx: %i and trans: %i"\
                     % (prop, value[0, 0], idx[0], t[0])
            ut.infoPrint(string)



    def setEint(self, idx, e_int, translation = 0, version = "lammps", verbose = 1):
        """Function for setting the interfacial energies of different translations

           idx - int representing interface index, 0-based.

           translation - int representing the specific translation, 0-based.

           e_int - float representing the energy value"""
        
        if version.lower() == "lammps":
            s = self.e_int.shape
        elif version.lower() == "vasp":
            s = self.e_int_vasp.shape
        t = translation

        if (t + 1) > s[1]:
            if version.lower() == "lammps":
                self.e_int = np.concatenate((self.e_int, np.zeros((s[0], (t + 1) - s[1]))), axis = 1)
            elif version.lower() == "vasp":
                self.e_int_vasp = np.concatenate((self.e_int_vasp, np.zeros((s[0], (t + 1) - s[1]))), axis = 1)

            if verbose > 0:
                string = "Extending e_int from shape (%i,%i) to (%i,%i)"\
                         % (s[0], s[1], s[0], (t + 1))
                ut.infoPrint(string)
        
        if verbose > 0:
            string = "E_int set to: %.2f at translation: %i for interface: %i"\
                     % (e_int, t, idx)
            ut.infoPrint(string)

        if version.lower() == "lammps":
            self.e_int[idx, t] = e_int
        elif version.lower() == "vasp":
            self.e_int_vasp[idx, t] = e_int



    def setEintArray(self, idx, translation, e_int, verbose = 1, version = "lammps"):
        """Function for setting the interfacial energies from an array of values

           idx - 1d vector of interface index, 0 based.

           translation - 1d vector of column index, 0 based.

           e_int - 2d array of shape (idx,translation) containing energies."""

        """Check inputs"""
        if not isinstance(translation, (np.ndarray, list, range)):
            print("<translation> parameter must be a np.ndarray, list or range")
            return
        if not isinstance(idx, (np.ndarray, list, range)):
            print("<idx> parameter must be a np.ndarray, list or range")
            return
        
        if version.lower() == "lammps":
            s = self.e_int.shape
        elif version.lower() == "vasp":
            s = self.e_int_vasp.shape
        t = translation

        if np.max(t) > s[1]:
            if version.lower() == "lammps":
                self.e_int = np.concatenate((self.e_int, np.zeros((s[0], (np.max(t) + 1) - s[1]))), axis = 1)
            elif version.lower() == "vasp":
                self.e_int_vasp = np.concatenate((self.e_int_vasp, np.zeros((s[0], (np.max(t) + 1) - s[1]))), axis = 1)

            if verbose > 0:
                string = "Extending e_int from shape (%i,%i) to (%i,%i)"\
                         % (s[0], s[1], s[0], np.max(t))
                ut.infoPrint(string)

        """t 0-based!"""
        for i, col in enumerate(t):
            if version.lower() == "lammps":
                self.e_int[idx, col] = e_int[:, i]
            elif version.lower() == "vasp":
                self.e_int_vasp[idx, col] = e_int[:, i]



    def setWsep(self, idx, w_sep, translation = 0, version = "lammps", verbose = 1):
        """Function for setting the work of sepparation of different translations

           idx - int representing interface index, 0-based.

           translation - int representing the specific translation, 0-based.

           w_sep - float representing the energy value"""
        
        if version.lower() == "lammps":
            s = self.w_sep.shape
        elif version.lower() == "vasp":
            s = self.w_sep_vasp.shape
        t = translation

        if (t + 1) > s[1]:
            if version.lower() == "lammps":
                self.w_sep = np.concatenate((self.w_sep, np.zeros((s[0], (t + 1) - s[1]))), axis = 1)
            elif version.lower() == "vasp":
                self.w_sep_vasp = np.concatenate((self.w_sep_vasp, np.zeros((s[0], (t + 1) - s[1]))), axis = 1)
            if verbose > 0:
                string = "Extending w_sep from shape (%i,%i) to (%i,%i)"\
                         % (s[0], s[1], s[0], (t + 1))
                ut.infoPrint(string)
        
        if verbose > 0:
            string = "W_sep set to: %.2f at translation: %i for interface: %i"\
                     % (w_sep, t, idx)
            ut.infoPrint(string)

        if version.lower() == "lammps":
            self.w_sep[idx, t] = w_sep
        elif version.lower() == "vasp":
            self.w_sep_vasp[idx, t] = w_sep



    def setWsepArray(self, idx, translation, w_sep, version = "lammps", verbose = 1):
        """Function for setting the work of sepparation from an array of values

           idx - 1d vector of interface index, 0 based.

           translation - 1d vector of column index, 0 based.

           w_sep - 2d array of shape (idx,translation) containing energies."""

        """Check inputs"""
        if not isinstance(translation, (np.ndarray, list, range)):
            print("<translation> parameter must be a np.ndarray, list or range")
            return
        if not isinstance(idx, (np.ndarray, list, range)):
            print("<idx> parameter must be a np.ndarray, list or range")
            return
        
        if version.lower() == "lammps":
            s = self.w_sep.shape
        elif version.lower() == "vasp":
            s = self.w_sep_vasp.shape
        t = translation

        if np.max(t) > s[1]:
            if version.lower() == "lammps":
                self.w_sep = np.concatenate((self.w_sep, np.zeros((s[0], (np.max(t) + 1) - s[1]))), axis = 1)
            elif version.lower() == "vasp":
                self.w_sep_vasp = np.concatenate((self.w_sep_vasp, np.zeros((s[0], (np.max(t) + 1) - s[1]))), axis = 1)
            if verbose > 0:
                string = "Extending w_sep from shape (%i,%i) to (%i,%i)"\
                         % (s[0], s[1], s[0], np.max(t))
                ut.infoPrint(string)

        """t 0-based!"""
        for i, col in enumerate(t):
            if version.lower() == "lammps":
                self.w_sep[idx, col] = w_sep[:, i]
            elif version.lower() == "vasp":
                self.w_sep_vasp[idx, col] = w_sep[:, i]



    def setWsepStrain(self, idx, w_sep_strain, translation = 0, version = "lammps", verbose = 1):
        """Function for setting the work of sepparation (strained ref) of different translations

           idx - int representing interface index, 0-based.

           translation - int representing the specific translation, 0-based.

           w_sep_strain - float representing the energy value"""
        
        if version.lower() == "lammps":
            s = self.w_sep_strain.shape
        elif version.lower() == "vasp":
            s = self.w_sep_strain_vasp.shape
        t = translation

        if (t + 1) > s[1]:
            if version.lower() == "lammps":
                self.w_sep_strain = np.concatenate((self.w_sep_strain,\
                                                    np.zeros((s[0], (t + 1) - s[1]))), axis = 1)
            elif version.lower() == "vasp":
                self.w_sep_strain_vasp = np.concatenate((self.w_sep_strain_vasp,\
                                                         np.zeros((s[0], (t + 1) - s[1]))), axis = 1)
            if verbose > 0:
                string = "Extending w_sep_strain from shape (%i,%i) to (%i,%i)"\
                         % (s[0], s[1], s[0], (t + 1))
                ut.infoPrint(string)
        
        if verbose > 0:
            string = "W_sep_strain set to: %.2f at translation: %i for interface: %i"\
                     % (w_sep, t, idx)
            ut.infoPrint(string)

        if version.lower() == "lammps":
            self.w_sep_strain[idx, t] = w_sep_strain
        elif version.lower() == "vasp":
            self.w_sep_strain_vasp[idx, t] = w_sep_strain



    def setWsepStrainArray(self, idx, translation, w_sep_strain, version = "lammps", verbose = 1):
        """Function for setting the work of sepparation from an array of values

           idx - 1d vector of interface index, 0 based.

           translation - 1d vector of column index, 0 based.

           w_sep_strain - 2d array of shape (idx,translation) containing energies."""

        """Check inputs"""
        if not isinstance(translation, (np.ndarray, list, range)):
            print("<translation> parameter must be a np.ndarray, list or range")
            return
        if not isinstance(idx, (np.ndarray, list, range)):
            print("<idx> parameter must be a np.ndarray, list or range")
            return
        
        if version.lower() == "lammps":
            s = self.w_sep_strain.shape
        elif version.lower() == "vasp":
            s = self.w_sep_strain_vasp.shape
        t = translation

        if np.max(t) > s[1]:
            if version.lower() == "lammps":
                self.w_sep_strain = np.concatenate((self.w_sep_strain,\
                                                    np.zeros((s[0], (np.max(t) + 1) - s[1]))), axis = 1)
            elif version.lower() == "vasp":
                self.w_sep_strain_vasp = np.concatenate((self.w_sep_strain_vasp,\
                                                         np.zeros((s[0], (np.max(t) + 1) - s[1]))), axis = 1)
            if verbose > 0:
                string = "Extending w_sep_strain from shape (%i,%i) to (%i,%i)"\
                         % (s[0], s[1], s[0], np.max(t))
                ut.infoPrint(string)

        """t 0-based!"""
        for i, col in enumerate(t):
            if version.lower() == "lammps":
                self.w_sep_strain[idx, col] = w_sep_strain[:, i]
            elif version.lower() == "vasp":
                self.w_sep_strain_vasp[idx, col] = w_sep_strain[:, i]


    def setAltBase(self, base = None, from_input = None, from_file = None, format = None):
        """Function for adding a new base"""

        """Read data from inputs or rom file if specified"""
        if from_input is not None: 
            base = inputs.getInputs(from_input)[0]
        elif from_file is not None:
            base = file_io.readData(filename = from_file, format = format)[0]

        if base is None:
            string = "No base or input specified. No alt_base set"
            ut.infoPrint(string)
            return

        original = len(self.alt_base)
        self.alt_base.append(base)
        string = "Added base [[%.5f %.5f %.5f],[%.5f %.5f %.5f],[%.5f %.5f %.5f]] as alt_base: %i"\
                 % (base[0, 0], base[0, 1], base[0, 2], base[1, 0], base[1, 1], base[1, 2],\
                    base[2, 0], base[2, 1], base[2, 2], original + 1)
        ut.infoPrint(string)
        


    def redefineCell(self, original = True, base = 0):
        """Redefines all parameters based on new parameters for the bottom cell"""
        print("Redefine Cell")
        


    def getAreas(self, idx = None, cell = 1, base = None):
        """Function for getting the area of multiple interfaces"""

        if idx is None: idx = np.arange(self.atoms.shape[0])

        if cell == 1:
            return np.abs(np.linalg.det(self.cell_1[idx, :, :]))
        elif cell == 2:
            return np.abs(np.linalg.det(self.cell_2[idx, :, :]))



    def getArea(self, idx, cell = 1):
        """Function for getting the area of the specified interface and surface"""

        if cell == 1:
            return np.abs(np.cross(self.cell_1[idx, :, 0], self.cell_1[idx, :, 1]))
        elif cell == 2:
            return np.abs(np.cross(self.cell_2[idx, :, 0], self.cell_2[idx, :, 1]))



    def getCellLengths(self, idx, cell = 1):
        """Function for getting the cell lengths of specified interfaces"""

        if not isinstance(idx, (np.ndarray, range, list)):
            idx = [idx]

        if cell == 1:
            norms = np.linalg.norm(self.cell_1[idx, :, :], axis = 1)
        elif cell == 2:
            norms = np.linalg.norm(self.cell_1[idx, :, :], axis = 1)

        return norms



    def getBaseAngle(self, idx, cell = 1, rad = True):
        """Function for getting the angle of the specified interface and surface"""

        if cell == 1:
            ang = np.arccos(np.dot(self.cell_1[idx, :, 0], self.cell_1[idx, :, 1]) /\
                            np.prod(np.linalg.norm(self.cell_1[idx, :, :], axis = 0)))

        elif cell == 2:
            ang = np.arccos(np.dot(self.cell_2[idx, :, 0], self.cell_2[idx, :, 1]) /\
                            np.prod(np.linalg.norm(self.cell_2[idx, :, :], axis = 0)))

        if not rad: 
            ang = np.rad2deg(ang)

        return ang



    def getBaseAngles(self, cell, idx = None, rad = True):
        """Function for getting the base angles of the bottom or top cells"""

        if idx is None: idx = np.arange(self.atoms.shape[0])
        if not isinstance(idx, (np.ndarray, range, list)):
            string = "Idx must be np.ndarray, range or list of index"
            ut.infoPrint(string)
            return

        if cell == 1:
            ang = np.arccos(np.sum(self.cell_1[idx, :, 0] * self.cell_1[idx, :, 1], axis = 1) /\
                            np.prod(np.linalg.norm(self.cell_1[idx, :, :], axis = 1), 1))
        elif cell == 2:
            ang = np.arccos(np.sum(self.cell_2[idx, :, 0] * self.cell_2[idx, :, 1], axis = 1) /\
                            np.prod(np.linalg.norm(self.cell_2[idx, :, :], axis = 1), 1))

        if not rad:
            ang = np.rad2deg(ang)

        return ang



    def getBaseLength(self, idx, cell = 1):
        """Function for getting the cell lengths of the specified interface and surface"""

        if cell == 1:
            l = np.linalg.norm(self.cell_1[idx, :, :], axis = 0)
        elif cell == 2:
            l = np.linalg.norm(self.cell_2[idx, :, :], axis = 0)

        return l



    def getBaseLengths(self, cell):
        """Function for getting the cell lengths"""

        if cell == 1:
            l = np.linalg.norm(self.cell_1, axis = 1)
        elif cell == 2:
            l = np.linalg.norm(self.cell_2, axis = 1)

        return l



    def matchCells(self, dTheta = 4, theta = None, n_max = 4, N = None,\
                   m_max = 4, M = None, max_strain = 1, max_atoms = 5000,\
                   limit = None, exp = 1, verbose = 1, min_angle = 10,\
                   remove_asd = True, asd_tol = 7, limit_asr = False,\
                   asr_tol = 1e-7, asr_iter = 350, asr_strain = "eps_mas",\
                   asr_endpoint = "over", target = None):

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
           N takes president as a specific range of repititions"""
        if N is None:
            nR = np.arange(-n_max, n_max + 1)

        """Repetions of the second cell vector, [0,...,m_max],
           M takes president as a specific range of repititions"""
        if M is None:
            mR = np.arange(0, m_max + 1)

        """Create all permutations of nR and mR if M,N is specifed use only those"""
        if M is not None and N is not None:
            M = np.array(M)[:, None]; N = np.array(N)[:, None]
            dPerm = np.concatenate((M, N), axis = 1)
        else:
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

        """If target is supplied the matching is done against
           those specific repetitions. Supplied as a 2x2 matrix 
           with basis vectors as columns. The righthanded version
           will be returned"""
        if target is not None:
            e = np.tile(np.array(target)[None, :, :], (R.shape[0], 1, 1))

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
            string = "Total basis pairs: %.0f | Lin dep/left handed: %.0f | Total kept: %.0f"\
                     % (keep.shape[0], keep.shape[0] - np.sum(keep), np.sum(keep))
            ut.infoPrint(string)

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
            string = "Non unique matches: %i | Total matches keept: %i"\
                      % (atoms.shape[0] - np.sum(keep), np.sum(keep))
            ut.infoPrint(string)

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
        self.e_int_c = np.zeros((self.atoms.shape[0], 1))
        self.w_sep_c = np.zeros((self.atoms.shape[0], 1))
        self.w_seps_c = np.zeros((self.atoms.shape[0], 1))
        self.e_int_d = np.zeros((self.atoms.shape[0], 1))
        self.w_sep_d = np.zeros((self.atoms.shape[0], 1))
        self.w_seps_d = np.zeros((self.atoms.shape[0], 1))

        """Further removal of interfaces based on specified critera follows below"""

        """Reject interfaces based on criteria of strain * atoms^exp > limit"""
        if limit is not None:
            keep = ((self.eps_mas * (self.atoms ** exp)) < limit)
            ratio = np.sum(np.logical_not(keep))
            if verbose > 0:
                string = "Matches with (strain * atoms^%s) > %s: %i | Total matches kept: %i"\
                         % (exp, limit, ratio, np.sum(keep))
                ut.infoPrint(string)

            """Remove interfaces with strain*atoms^exp > limit"""
            self.deleteInterfaces(keep, verbose = verbose - 1)

        """Remove cells with to narrow cell angles, defined below"""
        ang_lim = np.deg2rad(min_angle)
        ang_1 = self.getBaseAngles(cell = 1)
        ang_2 = self.getBaseAngles(cell = 2)

        keep = (ang_1 > ang_lim) * (ang_1 < np.pi - ang_lim) *\
               (ang_2 > ang_lim) * (ang_2 < np.pi - ang_lim)

        max_angle = np.sum(np.logical_not(keep))
        if verbose > 0:
            string = "Cell angle outside limit (%.1f<X<%.1f): %i | Total kept: %i"\
                     % (np.rad2deg(ang_lim), np.rad2deg(np.pi - ang_lim), max_angle, np.sum(keep))
            ut.infoPrint(string)

        """Remove interfaces with angles outside specified range"""
        self.deleteInterfaces(keep, verbose = verbose - 1)

        """Remove matches were any strain component is > max_strain"""
        keep = (np.abs(self.eps_11) < max_strain) *\
               (np.abs(self.eps_22) < max_strain) *\
               (np.abs(self.eps_12) < max_strain)

        max_strain = np.sum(np.logical_not(keep))
        if verbose > 0:
            string = "Matches above max strain: %i | Total matches kept: %i"\
                     % (max_strain, np.sum(keep))
            ut.infoPrint(string)

        """Remove interfaces with abs(strains) above max_strain"""
        self.deleteInterfaces(keep, verbose = verbose - 1)

        """Remove matches with the number of atoms > max_atoms"""
        keep = (self.atoms < max_atoms)
        max_atoms = np.sum(np.logical_not(keep))
        if verbose > 0:
            string = "Matches with to many atoms: %i | Total matches kept: %i"\
                     % (max_atoms, np.sum(keep))
            ut.infoPrint(string)

        """Remove interfaces with more atoms than max_atoms"""
        self.deleteInterfaces(keep, verbose = verbose - 1)

        """Find duplicates in the combo (nr_atoms, eps_mas) if specified"""
        if remove_asd:
            keep = self.getAtomStrainDuplicates(tol_mag = asd_tol, verbose = 0)
            if verbose > 0:
                string = "Duplicate atoms/strain combinations: %i | Total matches kept: %i"\
                         % (np.sum(np.logical_not(keep)), np.sum(keep))
                ut.infoPrint(string)

            """Remove duplicates"""
            self.removeAtomStrainDuplicates(tol_mag = asd_tol, verbose = verbose - 1)

        """Interfaces with |strains| < tol are slightly perturbed to avoid issues with log expressions"""
        tol = 1e-7
        exact_matches = np.abs(self.eps_mas) < tol
        self.eps_11[np.abs(self.eps_11) < tol] = tol
        self.eps_22[np.abs(self.eps_22) < tol] = tol
        self.eps_12[np.abs(self.eps_12) < tol] = tol
        self.eps_mas[np.abs(self.eps_mas) < tol] = tol
        if np.sum(exact_matches) > 0:
            string = "Exact matches found: %i" % np.sum(exact_matches)
            ut.infoPrint(string)

        """Remove interfaces based on atom strain ratios, limiting the set to this number"""
        if limit_asr and self.atoms.shape[0] > 2:
            self.removeByAtomStrain(keep = limit_asr, tol = asr_tol, max_iter = asr_iter,\
                                    strain = asr_strain, endpoint = asr_endpoint,\
                                    verbose = verbose)

        """Sort the interfaces based on number of atoms"""
        self.sortInterfaces()



    def printTranslation(self, surface, translation = None, verbose = 1):
        """Function for printing the specified translations"""

        if translation is None: translation = range(ut.getNrTranslations(surface = surface))
        if isinstance(translation, (int, np.integer)): translation = [translation]

        for t in translation:
            trans, site = ut.getTranslation(surface = surface, translation = t, verbose = verbose - 1)
            
            string = "Translation: %i, x: %5.2f, y: %5.2f, Site name: %s" % (t, trans[0], trans[1], site) 
            ut.infoPrint(string)


    def plotTranslations(self, surface, translation = None):
        """Function for plotting specified translations"""
        print("Do something")

        



    def printInterfaces(self, idx = None, sort = None, rev = False, flag = None, anchor = ""):
        """Print info about found interfaces"""

        if idx is None:
            idx = range(self.atoms.shape[0])
        elif isinstance(idx, (int, np.integer)):
            idx = [idx]

        if sort is not None:
            self.sortInterface(sort = sort, rev = rev)

        header1 = "%-2s%6s | %3s %-9s | %5s | %5s |  %-5s | %-5s | %4s %-25s | %3s %-11s | %3s %-11s "\
                  % ("", "Index", "", "Length", "Angle", "Angle", "Area", "Atoms", "", "Epsilon (*100)",\
                     "", "Lattice 1", "", "Lattice 2")
        header2 = "%-2s%6s | %6s, %5s | %5s | %5s | %6s | %5s | %7s,%7s,%7s,%6s | "\
                  "%3s,%3s,%3s,%3s | %3s,%3s,%3s,%3s"\
                  % ("", "i", "a1", "a2", "b1/b2", "a1/a2", "Ang^2", "Nr", "11", "22", "12", "mas",\
                     "a1x", "a1y", "a2x", "a2y", "b1x", "b1y", "b2x", "b2y")

        div = "=" * len(header1)
        print("\n" + header1 + "\n" + header2 + "\n" + div)

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

            at = self.atoms[i]

            if np.isin(i, flag):
                string = "%-2s%6.0f * %6.1f,%6.1f * %5.1f * %5.1f * %6.1f * %5.0f * "\
                    "%7.2f,%7.2f,%7.2f,%6.2f * %3i,%3i,%3i,%3i * %3i,%3i,%3i,%3i"\
                    % (anchor, i, la[0], la[1], ba, aa, ar, at, s1, s2, s3, s4,\
                           ra[0], ra[2], ra[1], ra[3], rb[0], rb[2], rb[1], rb[3])
            else:
                string = "%-2s%6.0f | %6.1f,%6.1f | %5.1f | %5.1f | %6.1f | %5.0f | "\
                    "%7.2f,%7.2f,%7.2f,%6.2f | %3i,%3i,%3i,%3i | %3i,%3i,%3i,%3i"\
                    % (anchor, i, la[0], la[1], ba, aa, ar, at, s1, s2, s3, s4,\
                           ra[0], ra[2], ra[1], ra[3], rb[0], rb[2], rb[1], rb[3])

            print(string)

        print(div + "\n")



    def summarize(self, idx, verbose = 1, save = False, format = "pdf",\
                  dpi = 100):
        """Plot a summary of information for the specified interface"""

        hFig = plt.figure(figsize = (8, 6))

        """First plot, base view"""
        self.plotInterface(annotate = True, idx = idx, verbose = verbose,\
                           align_base = "no", scale = False, save = False,\
                           handle = True, col = 2, row = 2, N = 1)

        hAx = plt.gca()
        hAx.set_xlabel("x, ($\AA$)")
        hAx.set_ylabel("y, ($\AA$)")
        hAx.tick_params()

        """Second plot, align cell 1 to x axis"""
        self.plotInterface(annotate = True, idx = idx, verbose = 0,\
                           align_base = "cell_1", scale = False, save = False,\
                           handle = True, col = 2, row = 2, N = 2)

        hAx = plt.gca()
        hAx.yaxis.tick_right()
        hAx.yaxis.set_label_position("right")
        hAx.set_xlabel(r"x, ($\AA$)")
        hAx.set_ylabel(r"y, ($\AA$)")

        """Third plot, all interface combinations, mark this"""
        C, E = self.getAtomStrainExpression(verbose = verbose - 1)
        self.plotCombinations(const = C, exp = E, mark = idx, save = False, handle = True,\
                              eps = "eps_mas", verbose = verbose - 1, col = 2, row = 2, N = 3,\
                              mark_ms = 5, mark_m = "x")
        hAx = plt.gca()
        hAx.set_xscale("log")
        hAx.set_yscale("log")
        hAx.set_xlabel("$\epsilon_{mas}$, (%)")
        hAx.set_ylabel("Atoms")

        self.hexPlotCombinations(eps = "eps_mas", col = 2, row = 2, N = 4,\
                                 verbose = verbose - 1, handle = True)
        hAx = plt.gca()
        hAx.yaxis.tick_right()
        hAx.yaxis.set_label_position("right")
        hAx.set_xlabel("$\epsilon_{mas}$, (%)")
        hAx.set_ylabel("Atoms")

        plt.tight_layout(h_pad = 0.2, w_pad = 2)
        if save:
            if save is True:
                ut.save_fig(filename = "Summary_%s.%s" % (idx, format), format = format,\
                         dpi = dpi, verbose = verbose)
            else:
                ut.save_fig(filename = save, format = format, dpi = dpi,\
                         verbose = verbose)
            plt.close()
        else:
            plt.show()


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
                ut.save_fig(filename = "interface%s.%s" % (add, format), format = format,\
                         dpi = dpi, verbose = verbose)
            else:
                ut.save_fig(filename = save, format = format, dpi = dpi,\
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
            hAx.set_ylabel(r"y, ($\AA$)")
            hAx.set_xlabel(r"x, ($\AA$)")

            hAx.legend(framealpha = 1)
        else:
            if np.isin(N, range(1, row*col + 1, col)):
                hAx.set_ylabel("y, ($\AA$)")
            if np.isin(N, range((row - 1) * col + 1, row * col + 1)):
                hAx.set_xlabel("x, ($\AA$)")

        if handle: return

        plt.tight_layout()
        if save:
            if save is True:
                ut.save_fig(filename = "interface_%s.%s" % (idx, format),\
                         format = format, dpi = dpi, verbose = verbose)
            else:
                ut.save_fig(filename = save, format = format, dpi = dpi,\
                         verbose = verbose)
            plt.close()
        else:
            plt.show()


    def hexPlotCombinations(self, idx = None, eps = "eps_mas",\
                            save = False, format = "pdf", dpi = 100,\
                            col = 1, row = 1, N = 1, handle = False,\
                            verbose = 1, **kwarg):
        """Hexplot of specified combinations"""

        if idx is None: idx = np.arange(self.atoms.shape[0])

        if not handle: hFig = plt.figure()
        hAx = plt.subplot(row, col, N)

        """Defaults"""
        cm = kwarg.pop("cmap", "plasma")
        
        if eps == "eps_11":
            hb = hAx.hexbin(np.abs(self.eps_11[idx]) * 100, self.atoms[idx], cmap = cm,\
                            xscale = "log", yscale = "log", mincnt = 1, **kwarg)
            x_label = "Strain $abs(\epsilon_{11})$, (%)"

        elif eps == "eps_22":
            hb = hAx.hexbin(np.abs(self.eps_22[idx]) * 100, self.atoms[idx], cmap = cm,\
                            xscale = "log", yscale = "log", mincnt = 1, **kwarg)
            x_label = "Strain $abs(\epsilon_{22})$, (%)"

        elif eps == "eps_12":
            hb = hAx.hexbin(np.abs(self.eps_12[idx]) * 100, self.atoms[idx], cmap = cm,\
                            xscale = "log", yscale = "log", mincnt = 1, **kwarg)
            x_label = "Strain $abs(\epsilon_{12})$, (%)"

        else:
            hb = hAx.hexbin(self.eps_mas[idx] * 100, self.atoms[idx], cmap = cm,\
                            xscale = "log", yscale = "log", mincnt = 1, **kwarg)
            x_label = "Strain $\epsilon_{mas}$, (%)"

        if handle: return

        if verbose > 0 and eps != "eps_mas":
            string = "Showing absolute value of %s" % (eps)
            ut.infoPrint(string)

        hAx.set_xlabel(x_label)
        hAx.set_ylabel("Atoms")
        cb = plt.colorbar(hb, ax = hAx)
        cb.set_label("Counts")

        if verbose > 0:
            string = "Total items: %i" % idx.shape[0]
            ut.infoPrint(string)

        hAx.set_title(self.filename)
        plt.tight_layout()
        if save:
            if save is True:
                ut.save_fig(filename = "Hexbin.%s" % format, format = format,\
                         dpi = dpi, verbose = verbose)
            else:
                ut.save_fig(filename = save, format = format, dpi = dpi,\
                         verbose = verbose)
            plt.close()
        else:
            plt.show()



    def plotCombinations(self, idx = None, const = None, exp = 1,\
                         mark = None, save = False, format = "pdf",\
                         dpi = 100, handle = False, eps = "eps_mas",\
                         verbose = 1, col = 1, row = 1, N = 1, mark_ms = 3,\
                         mark_m = "o", marker = "o", **kwarg):
        """Plots strain vs. atoms for the interfaces"""

        if idx is None: idx = np.arange(self.atoms.shape[0])

        if not handle: hFig = plt.figure()
        hAx = plt.subplot(row, col, N)
        
        atoms = self.atoms[idx]

        if eps.lower() == "eps_11":
            strain = np.abs(self.eps_11)[idx]
            x_label = "$|\epsilon_{11}|$, (%)"
            if verbose > 0: print("Showing absolute value of %s" % (eps))
        elif eps.lower() == "eps_22":
            strain = np.abs(self.eps_22)[idx]
            x_label = "$|\epsilon_{22}|$, (%)"
            if verbose > 0: print("Showing absolute value of %s" % (eps))
        elif eps.lower() == "eps_12":
            strain = np.abs(self.eps_12)[idx]
            x_label = "$|\epsilon_{12}|$, (%)"
            if verbose > 0: print("Showing absolute value of %s" % (eps))
        else:
            x_label = "$(\epsilon_{11}+\epsilon_{22}+\epsilon_{12})/3$, (%)"
            strain = self.eps_mas[idx]

        if verbose > 0:
            string = "Items total: %i" % (np.shape(idx)[0])
            ut.infoPrint(string)

        if mark is not None:
            if isinstance(mark, (int, np.integer)) or isinstance(mark[0], (int, np.integer)):
                mask = np.zeros(np.shape(idx), dtype = bool)
                mask[mark] = True
            else:
                mask = mark

            strain_m = strain[mask]
            atoms_m = atoms[mask]
            strain = strain[np.logical_not(mask)]
            atoms = atoms[np.logical_not(mask)]


        lines = []
        if const is not None and self.atoms.shape[0] > 2:

            """Find atom/strain pairs below limit set by atoms = A * strain ** exp"""
            low = atoms < (const * strain ** exp)
            hi = np.logical_not(low)

            l = hAx.plot(strain[low] * 100, atoms[low], color = 'b', linestyle = "None",\
                              marker = "o", mew = 0.5, **kwarg)

            lines.append(l[0])

            l = hAx.plot(strain[hi] * 100, atoms[hi], color = 'r', linestyle = "None",\
                         marker = "o", mew = 0.5, **kwarg)

            lines.append(l[0])

            """Plot the dividing line for the specified limit"""
            j = np.log(np.max(atoms) / const) / exp
            k = np.log(np.min(atoms) / const) / exp
            x = np.logspace(j, k, 1000, base = np.exp(1))

            hAx.plot(x * 100, const * x ** exp, linewidth = 0.5, color = 'k')

            if verbose > 0:
                string = "Items below: %i | Items above: %i" % (np.sum(low), np.sum(hi))
                ut.infoPrint(string)
        else:
            l = hAx.plot(strain * 100, atoms, color = 'b', linestyle = "None", marker = "o",\
                         mew = 0.5, **kwarg)

            lines.append(l[0])

        if mark is not None:
            l = hAx.plot(strain_m * 100, atoms_m, color = 'k', marker = mark_m,\
                         linestyle = "None", markersize = mark_ms, mfc = 'k')
            
            lines.append(l[0])

            if verbose > 0:
                string = "Items marked: %i" % (strain_m.shape[0])
                ut.infoPrint(string)
                if verbose > 1:
                    self.printInterfaces(idx = np.arange(mask.shape[0])[mask])
        
        if handle: return

        for line in lines:
            line.set_picker(3)

        anP = hAx.plot([], [], marker = 'o', ms = 5, color = 'm', mew = 2, mfc = 'None')

        def update_annotation(line, ind, event):
            x, y = line.get_data()
            closest = np.argmin(np.sqrt((x[ind["ind"]] - event.xdata)**2 +\
                                        (y[ind["ind"]] - event.ydata)**2))
            xSel = x[ind["ind"]][closest]
            ySel = y[ind["ind"]][closest]

            dx = 1e-7
            dy = 1e-7

            ms = plt.getp(line, "markersize") * 2.7
            plt.setp(anP[0], markersize = ms)
            anP[0].set_data(xSel, ySel)

            S = self.getStrain(idx = idx, strain = eps, base_1 = None, base_2 = None)
            match = (np.abs(S - xSel / 100) < dx) * (np.abs(self.atoms[idx] - ySel) < dy)

            match = np.arange(match.shape[0])[match]
            self.printInterfaces(idx = match)


        def click(event):
            if event.inaxes == hAx:
                for line in lines:
                    cont, ind = line.contains(event)
                    if cont:
                        break

                if cont:
                    update_annotation(line, ind, event)
                    hFig.canvas.draw_idle()
                else:
                    anP[0].set_data([], [])
                    hFig.canvas.draw_idle()
            

        hAx.set_xscale("log")
        hAx.set_yscale("log")
        hAx.set_ylabel("Nr of Atoms")
        hAx.set_xlabel(x_label)
        hAx.set_title(self.filename)

        plt.tight_layout()
        if save:
            if save is True:
                ut.save_fig(filename = "Combinations.%s" % format, format = format,\
                         dpi = dpi, verbose = verbose)
            else:
                ut.save_fig(filename = save, format = format, dpi = dpi,\
                         verbose = verbose)
            plt.close()
        else:
            hFig.canvas.mpl_connect("button_release_event", click)
            plt.show()


            
    def getIndexOf(self, property = "eps_mas"):
        print("Function to grab index using criteria")
        print("To be built")



    def getProperties(self, var, label = "full"):
        """Function for retriving data from variables"""

        print("Get Properties")



    def plotProperty(self, x, y, z = None, idx = None, col = 1, row = 1, N = 1, 
                     save = False, dpi = 100, format = "pdf", verbose = 1, handle = False,\
                     translation = None, title = None, other = None, **kwargs):
        """Function for plotting properties agains each other.

        Available properties are
        ------------------------
        idx        = Index of current sorting
        eps_11     = Eps_11
        eps_22     = Eps_22
        eps_12     = Eps_12
        eps_mas    = Eps_mas
        eps_max    = max(eps_11, eps_22, eps_12)
        atoms      = Nr of atoms
        angle      = Angle between interface cell vectors
        rotation   = Initial rotation at creation
        norm       = Sqrt(eps_11^2+eps_22^2+eps_12^2)
        trace      = |eps_11|+|eps_22|
        norm_trace = Sqrt(eps_11^2+eps_22^2)
        a_1        = Length of interface cell vector a_1
        a_2        = Length of interface cell vector a_2
        area       = Area of the interface
        other      = Plot a custom array of values specified with keyword other. Length must match idx.
        e_int_c       = Interfacial energy, for specified translation(s)
        e_int_d       = Interfacial energy (vasp), for specified translation(s)
        e_int_diff_c  = Difference in Interfacial energy between translations
        e_int_diff_d  = Difference in Interfacial energy (vasp) between translations
        w_sep_c       = Work of separation, for specified translation(s)
        w_sep_d       = Work of separation (DFT), for specified translation(s)
        w_seps_c      = Work of separation (strained ref), for specified translation(s)
        w_seps_d      = Work of separation (strained ref) (DFT), for specified translation(s)
        w_sep_diff_c  = Difference in w_sep_c between tranlsations
        w_sep_diff_d  = Difference in w_sep_d (DFT) between tranlsations
        w_seps_diff_c = Difference in w_seps_c between translations
        w_seps_diff_d = Difference in w_seps_d (DFT) between translations

        plot x vs. y vs. z (optional) with z data values displayed in a colormap.
        """

        if idx is None: idx = np.arange(self.atoms.shape[0])
        if translation is None: translation = [0]
        if isinstance(translation, (int, np.integer)): translation = [translation]
        
        if save is True:
            filename = "%s_%s.%s" % (x, y, format)

        data = {"x": x, "y": y, "z": z}
        lbl = {"x": "", "y": "", "z": ""}
        raw_x = x; raw_y = y
        b1 = None; b2 = None

        for key in data:
            if data[key] is None: continue

            if data[key].lower() == "idx":
                data[key] = np.array(idx)
                lbl[key] = "Index"

            elif data[key].lower() == "eps_11":
                data[key] = self.getStrain(idx = idx, strain = "eps_11", base_1 = b1, base_2 = b2)
                lbl[key] = "$\epsilon_{11}$"

            elif data[key].lower() == "eps_22":
                data[key] = self.getStrain(idx = idx, strain = "eps_22", base_1 = b1, base_2 = b2)
                lbl[key] = "$\epsilon_{22}$"

            elif data[key].lower() == "eps_12":
                data[key] = self.getStrain(idx = idx, strain = "eps_12", base_1 = b1, base_2 = b2)
                lbl[key] = "$\epsilon_{12}$"

            elif data[key].lower() == "eps_mas":
                data[key] = self.getStrain(idx = idx, strain = "eps_mas", base_1 = b1, base_2 = b2)
                lbl[key] = "$\epsilon_{mas}$"

            elif data[key].lower() == "eps_max":
                eps_stack = self.getStrain(idx = idx, strain = "array",\
                                           base_1 = b1, base_2 = b2)
                max = np.max(eps_stack, axis = 0)
                min = np.min(eps_stack, axis = 0)
                """Check if abs(min) is bigger than max, (to preserve sign)"""
                max[np.abs(min) > np.abs(max)] = min[np.abs(min) > np.abs(max)]
                data[key] = max
                lbl[key] = "Max$(\epsilon_{11},\epsilon_{22},\epsilon_{12})$"

            elif data[key].lower() == "atoms":
                data[key] = self.atoms[idx]
                lbl[key] = "Atoms"

            elif data[key].lower() == "angle":
                data[key] = self.getBaseAngles(idx = idx, cell = 1, rad = False)
                lbl[key] = "Cell Angle, (Deg)"

            elif data[key].lower() == "norm":
                eps_stack = self.getStrain(idx = idx, strain = "array",\
                                           base_1 = b1, base_2 = b2)
                data[key] = np.linalg.norm(eps_stack, axis = 0)
                lbl[key] = "$\sqrt{\epsilon_{11}^2+\epsilon_{22}^2+\epsilon_{21}^2}$"

            elif data[key].lower() == "norm_trace":
                eps_stack = self.getStrain(idx = idx, strain = "array",\
                                           base_1 = b1, base_2 = b2)
                data[key] = np.sqrt(eps_stack[1, :]**2 + eps_stack[0, :]**2)
                lbl[key] = "$\sqrt{\epsilon_{11}^2+\epsilon_{22}^2}$"

            elif data[key].lower() == "trace":
                eps_stack = self.getStrain(idx = idx, strain = "array",\
                                           base_1 = b1, base_2 = b2)
                data[key] = np.abs(eps_stack[0, :]) + np.abs(eps_stack[1, :])
                lbl[key] = "$|\epsilon_{11}|+|\epsilon_{22}|$"

            elif data[key].lower() == "rotation":
                data[key] = self.ang[idx]
                lbl[key] = "Initial rotaion of top cell, (Deg)"

            elif data[key].lower() == "a_1":
                data[key] = self.getCellLengths(idx = idx, cell = 1)[:, 0]
                lbl[key] = "Length $a_1$, ($\AA$)"

            elif data[key].lower() == "a_2":
                data[key] = self.getCellLengths(idx = idx, cell = 1)[:, 1]
                lbl[key] = "Length $a_2$, ($\AA$)"

            elif data[key].lower() == "area":
                data[key] = self.getAreas(idx = idx, cell = 1)
                lbl[key] = "Area, ($\AA^2$)"

            elif data[key].lower() == "density":
                area = self.getAreas(idx = idx, cell = 1)
                base_area_1 = np.abs(np.cross(self.base_1[0, :2], self.base_1[1, :2]))
                base_area_2 = np.abs(np.cross(self.base_2[0, :2], self.base_2[1, :2]))

                vol = self.base_2[2, 2] * area
                atoms = self.atoms[idx] - area * (self.pos_1.shape[0] / base_area_1)
                norm_density = self.pos_2.shape[0] / (base_area_2 * self.base_2[2, 2])

                data[key] = atoms / (vol * norm_density)
                lbl[key] = "Atom density, ($Atoms/(\AA^2*standard)$)"

            elif data[key].lower() == "e_int_c":
                if len(translation) > self.e_int_c.shape[1]:
                    string = "Translation (%i) outside e_int_c range (0,%i)"\
                             % (np.max(translation), self.e_int_c.shape[1])
                    ut.infoPrint(string)
                    return

                data[key] = self.e_int_c[idx, :][:, translation]
                lbl[key] = "Interfacial Energy, ($eV/\AA2$)"

            elif data[key].lower() == "e_int_d":
                if len(translation) > self.e_int_d.shape[1]:
                    string = "Translation (%i) outside e_int_d range (0,%i)"\
                             % (np.max(translation), self.e_int_d.shape[1])
                    ut.infoPrint(string)
                    return

                data[key] = self.e_int_d[idx, :][:, translation]
                lbl[key] = "Interfacial Energy (DFT), ($eV/\AA2$)"

            elif data[key].lower() == "e_int_diff_c":
                if len(translation) > self.e_int_c.shape[1]:
                    string = "Translation (%i) outside e_int_c range (0,%i)"\
                             % (np.max(translation), self.e_int_c.shape[1])
                    ut.infoPrint(string)
                    return
                
                data[key] = np.max(self.e_int_c[idx, :][:, translation], axis = 1) -\
                            np.min(self.e_int_c[idx, :][:, translation], axis = 1)
                lbl[key] = "$\Delta$ in Interfacial Energy, ($eV/\AA^2$)"

            elif data[key].lower() == "e_int_diff_d":
                if len(translation) > self.e_int_d.shape[1]:
                    string = "Translation (%i) outside e_int_d range (0,%i)"\
                             % (np.max(translation), self.e_int_d.shape[1])
                    ut.infoPrint(string)
                    return
                
                data[key] = np.max(self.e_int_d[idx, :][:, translation], axis = 1) -\
                            np.min(self.e_int_d[idx, :][:, translation], axis = 1)
                lbl[key] = "$\Delta$ in Interfacial Energy (DFT), ($eV/\AA^2$)"

            elif data[key].lower() == "w_sep_c":
                if len(translation) > self.w_sep_c.shape[1]:
                    string = "Translation (%i) outside w_sep_c range (0,%i)"\
                             % (np.max(translation), self.w_sep_c.shape[1])
                    ut.infoPrint(string)
                    return
                
                data[key] = self.w_sep_c[idx, :][:, translation]
                lbl[key] = "Work of Separation, ($eV/\AA^2$)"

            elif data[key].lower() == "w_sep_d":
                if len(translation) > self.w_sep_d.shape[1]:
                    string = "Translation (%i) outside w_sep_d range (0,%i)"\
                             % (np.max(translation), self.w_sep_d.shape[1])
                    ut.infoPrint(string)
                    return
                
                data[key] = self.w_sep_d[idx, :][:, translation]
                lbl[key] = "Work of Separation (DFT), ($eV/\AA^2$)"

            elif data[key].lower() == "w_sep_diff_c":
                if len(translation) > self.w_sep_c.shape[1]:
                    string = "Translation (%i) outside w_sep_c range (0,%i)"\
                             % (np.max(translation), self.w_sep_c.shape[1])
                    ut.infoPrint(string)
                    return
                
                data[key] = np.max(self.w_sep_c[idx, :][:, translation], axis = 1) -\
                            np.min(self.w_sep_c[idx, :][:, translation], axis = 1)
                lbl[key] = "$\Delta$ in Work of Separation, ($eV/\AA^2$)"

            elif data[key].lower() == "w_sep_diff_d":
                if len(translation) > self.w_sep_d.shape[1]:
                    string = "Translation (%i) outside w_sep_d range (0,%i)"\
                             % (np.max(translation), self.w_sep_d.shape[1])
                    ut.infoPrint(string)
                    return
                
                data[key] = np.max(self.w_sep_d[idx, :][:, translation], axis = 1) -\
                            np.min(self.w_sep_d[idx, :][:, translation], axis = 1)
                lbl[key] = "Diff in Work of Separation (DFT), ($eV/\AA^2$)"

            elif data[key].lower() == "w_seps_diff_c":
                if len(translation) > self.w_seps_c.shape[1]:
                    string = "Translation (%i) outside w_seps_c range (0,%i)"\
                             % (np.max(translation), self.w_seps_c.shape[1])
                    ut.infoPrint(string)
                    return
                
                data[key] = np.max(self.w_seps_c[idx, :][:, translation], axis = 1) -\
                            np.min(self.w_seps_c[idx, :][:, translation], axis = 1)
                lbl[key] = "$\Delta$ in Work of Separation (strained), ($eV/\AA^2$)"

            elif data[key].lower() == "w_seps_diff_d":
                if len(translation) > self.w_seps_d.shape[1]:
                    string = "Translation (%i) outside w_seps_d range (0,%i)"\
                             % (np.max(translation), self.w_seps_d.shape[1])
                    ut.infoPrint(string)
                    return
                
                data[key] = np.max(self.w_seps_d[idx, :][:, translation], axis = 1) -\
                            np.min(self.w_seps_d[idx, :][:, translation], axis = 1)
                lbl[key] = "$\Delta$ in Work of Separation (strained) (DFT), ($eV/\AA^2$)"

            elif data[key].lower() == "w_seps_c":
                if len(translation) > self.w_seps_c.shape[1]:
                    string = "Translation (%i) outside w_seps_c range (0,%i)"\
                             % (np.max(translation), self.w_seps_c.shape[1])
                    ut.infoPrint(string)
                    return
                
                data[key] = self.w_seps_c[idx, :][:, translation]
                lbl[key] = "Work of Separation (strained), ($eV/\AA^2$)"

            elif data[key].lower() == "w_seps_d":
                if len(translation) > self.w_seps_d.shape[1]:
                    string = "Translation (%i) outside w_seps_d range (0,%i)"\
                             % (np.max(translation), self.w_seps_d.shape[1])
                    ut.infoPrint(string)
                    return
                
                data[key] = self.w_seps_d[idx, :][:, translation]
                lbl[key] = "Work of Separation (strained) (VASP), ($eV/\AA^2$)"

            elif data[key].lower() == "other":
                data[key] = other
                lbl[key] = "Custom"

            else:
                string = "Unrecognized key: %s" % key
                ut.infoPrint(string)
                return

        hFig = plt.figure()

        if data["z"] is None:
            ls = kwargs.pop("linestyle", "none")
            m = kwargs.pop("marker", "o")
            ms = kwargs.pop("markersize", 1.5)
            mew = kwargs.pop("markeredgewidth", 1)

            hAx = plt.subplot(row, col, N)
            hP = hAx.plot(data["x"], data["y"], linestyle = ls, mew = mew, marker = m,\
                          markersize = ms, **kwargs)
            hAx.set_xlabel(lbl["x"])
            hAx.set_ylabel(lbl["y"])
        else:
            hAx = plt.subplot(row, col, N)

            if np.ndim(data["x"]) == 1: data["x"] = data["x"][:, None]
            if np.ndim(data["y"]) == 1: data["y"] = data["y"][:, None]
            if np.ndim(data["z"]) == 1: data["z"] = data["z"][:, None]

            cm = kwargs.pop("colormap", "viridis")
            cmap = plt.cm.get_cmap(cm)
            vmin = kwargs.pop("vmin", np.min(data["z"]))
            vmax = kwargs.pop("vmax", np.max(data["z"]))
            c = kwargs.pop("color", 'b')

            hP = []
            j,k,l = (0, 0, 0)
            for i, t in enumerate(translation):
            
                tP = hAx.scatter(data["x"][:, j], data["y"][:, k], c = data["z"][:, l],\
                                 vmin = vmin, vmax = vmax, cmap = cmap, **kwargs)

                hP.append(tP)

                if np.shape(data["x"])[1] > 1: j += 1
                if np.shape(data["y"])[1] > 1: k += 1
                if np.shape(data["z"])[1] > 1: l += 1
                
            plt.colorbar(hP[0], label = lbl["z"])
            hAx.set_xlabel(lbl["x"])
            hAx.set_ylabel(lbl["y"])

        if handle: return

        if title is None:
            hAx.set_title(self.filename)
        else:
            hAx.set_title(title)

        if len(translation) > 1:
            lgd = []
            for i in translation:
                lgd.append("$T_{%i}$" % i)
                plt.legend(lgd, framealpha = 1)

        """Annotating plot marker"""
        hP[0].set_picker(2)
        anP = hAx.plot([], [], marker = 'o', ms = 6, color = 'k', mew = 2, mfc = 'None',\
                       linestyle = 'None')
        plt.tight_layout()

        """Function to allow clickable points to display information"""
        def click(event):
            if event.inaxes == hAx:

                for line in hP:
                    cont, ind = line.contains(event)
                    if cont:
                        break

                if cont:
                    if data["z"] is not None:
                        x = line.get_offsets()[:, 0]
                        y = line.get_offsets()[:, 1]
                    else:
                        x, y = line.get_data()

                    xSel = x[ind["ind"]]
                    ySel = y[ind["ind"]]

                    pPos = hAx.transData.transform((xSel, ySel))
                    pDist = np.linalg.norm(pPos - [[event.x, event.y]], axis = 1)
                    index = ind["ind"][np.argmin(pDist)]
                    anP[0].set_data(x[ind["ind"]], y[ind["ind"]])
                    for n, i in enumerate(ind["ind"]):
                        string = "Idx: %i  %s: %.4f  %s: %.4f  Nr Points: %i"\
                            % (idx[i], raw_x, x[i], raw_y, y[i], len(ind["ind"]))

                        if n == 0: 
                            print("=" * len(string))
                        print(string)
                        if n == len(ind["ind"]) - 1: 
                            print("=" * len(string))

                    hFig.canvas.draw_idle()
                else:
                    anP[0].set_data([], [])
                    hFig.canvas.draw_idle()


        if save:
            if save is True:
                ut.save_fig(filename = "PropertyPlot.%s" % format, format = format,\
                         dpi = dpi, verbose = verbose)
            else:
                ut.save_fig(filename = save, format = format, dpi = dpi,\
                         verbose = verbose)
            plt.close()
        else:
            hFig.canvas.mpl_connect("button_release_event", click)
            plt.show()



    def compareInterfaces(self, var, idx = None, translation = None, other = None,\
                          save = False, dpi = 100, format = "pdf", row = 1,\
                          col = 1, N = 1, handle = False, cmap = "tab10",\
                          m = 'o', ls = 'None', delta = False, verbose = 1):
        """Function for plotting comparisons of interface properties"""

        if idx is None: idx = np.arange(self.atoms.shape[0])
        if translation is None: translation = [0]
        if isinstance(translation, (int, np.integer)): translation = [translation]
        
        data, lbl = self.getData(var = var, idx = idx, translation = translation)

        hFig = plt.figure()
        hAx = plt.subplot(row, col, N)
        c = plt.cm.get_cmap(cmap)(range(len(data)))
        for i, key in enumerate(data):
            if delta:
                if i == 0:
                    d = data[key]
                else:
                    d = np.vstack((d, data[key]))
                
            hAx.plot(idx, data[key], label = lbl[key], color = c[i, :], marker = m, ls = ls)

        if delta:
            hAxr = hAx.twinx()
            hAxr.plot(idx, np.max(d, axis = 0) - np.min(d, axis = 0), label = "$\Delta$", ls = "--",\
                      c = [0, 0, 0, 0.45])
            hAxr.set_ylabel("$\Delta$")

        hAx.set_ylabel("Property")
        hAx.set_xlabel("Index")
        hAx.set_title("Comparison")
        hAx.legend(framealpha = 0.4)

        hFig.tight_layout()

        if save:
            if save is True:
                ut.save_fig(filename = "Comparison.%s" % format, format = format,\
                         dpi = dpi, verbose = verbose)
            else:
                ut.save_fig(filename = save, format = format, dpi = dpi,\
                         verbose = verbose)
            plt.close()
        else:
            plt.show()
            





    def getData(self, var, idx = None, translation = None):
        """Function for returning values for selected properties in an 
        array size [var, idx]
        
        Available values for "var" are
        ------------------------
        idx        = Index of current sorting
        eps_11     = Eps_11
        eps_22     = Eps_22
        eps_12     = Eps_12
        eps_mas    = Eps_mas
        eps_max    = max(eps_11, eps_22, eps_12)
        atoms      = Nr of atoms
        angle      = Angle between interface cell vectors
        rotation   = Initial rotation at creation
        norm       = Sqrt(eps_11^2+eps_22^2+eps_12^2)
        trace      = |eps_11|+|eps_22|
        norm_trace = Sqrt(eps_11^2+eps_22^2)
        a_1        = Length of interface cell vector a_1
        a_2        = Length of interface cell vector a_2
        area       = Area of the interface
        other      = Plot a custom array of values specified with keyword other. Length must match idx.
        e_int_c       = Interfacial energy, for specified translation(s)
        e_int_d       = Interfacial energy (DFT), for specified translation(s)
        e_int_diff_c  = Difference in Interfacial energy between translations
        e_int_diff_d  = Difference in Interfacial energy (DFT) between translations
        w_sep_c       = Work of separation, for specified translation(s)
        w_sep_d       = Work of separation (DFT), for specified translation(s)
        w_seps_c      = Work of separation (strained ref), for specified translation(s)
        w_seps_d      = Work of separation (strained ref) (DFT), for specified translation(s)
        w_sep_diff_c  = Difference in w_sep_c between tranlsations
        w_sep_diff_d  = Difference in w_sep_d (DFT) between tranlsations
        w_seps_diff_c = Difference in w_seps_c between translations
        w_seps_diff_d = Difference in w_seps_d (DFT) between translations
        """ 

        if idx is None: idx = np.arange(self.atoms.shape[0])
        if translation is None: translation = [0]
        if isinstance(translation, (int, np.integer)): translation = [translation]

        data = {}
        lbl = {}
        b1 = None; b2 = None

        for key in var:
            if key.lower() == "idx":
                data[key] = np.array(idx)
                lbl[key] = "Index"

            elif key.lower() == "eps_11":
                data[key] = self.getStrain(idx = idx, strain = "eps_11", base_1 = b1, base_2 = b2)
                lbl[key] = "$\epsilon_{11}$"

            elif key.lower() == "eps_22":
                data[key] = self.getStrain(idx = idx, strain = "eps_22", base_1 = b1, base_2 = b2)
                lbl[key] = "$\epsilon_{22}$"

            elif key.lower() == "eps_12":
                data[key] = self.getStrain(idx = idx, strain = "eps_12", base_1 = b1, base_2 = b2)
                lbl[key] = "$\epsilon_{12}$"

            elif key.lower() == "eps_mas":
                data[key] = self.getStrain(idx = idx, strain = "eps_mas", base_1 = b1, base_2 = b2)
                lbl[key] = "$\epsilon_{mas}$"

            elif key.lower() == "eps_max":
                eps_stack = self.getStrain(idx = idx, strain = "array",\
                                           base_1 = b1, base_2 = b2)
                max = np.max(eps_stack, axis = 0)
                min = np.min(eps_stack, axis = 0)
                """Check if abs(min) is bigger than max, (to preserve sign)"""
                max[np.abs(min) > np.abs(max)] = min[np.abs(min) > np.abs(max)]
                data[key] = max
                lbl[key] = "Max$(\epsilon_{11},\epsilon_{22},\epsilon_{12})$"

            elif key.lower() == "atoms":
                data[key] = self.atoms[idx]
                lbl[key] = "Atoms"

            elif key.lower() == "angle":
                data[key] = self.getBaseAngles(idx = idx, cell = 1, rad = False)
                lbl[key] = "Cell Ang, (Deg)"

            elif key.lower() == "norm":
                eps_stack = self.getStrain(idx = idx, strain = "array",\
                                           base_1 = b1, base_2 = b2)
                data[key] = np.linalg.norm(eps_stack, axis = 0)
                lbl[key] = "$\sqrt{\epsilon_{11}^2+\epsilon_{22}^2+\epsilon_{21}^2}$"

            elif key.lower() == "norm_trace":
                eps_stack = self.getStrain(idx = idx, strain = "array",\
                                           base_1 = b1, base_2 = b2)
                data[key] = np.sqrt(eps_stack[1, :]**2 + eps_stack[0, :]**2)
                lbl[key] = "$\sqrt{\epsilon_{11}^2+\epsilon_{22}^2}$"

            elif key.lower() == "trace":
                eps_stack = self.getStrain(idx = idx, strain = "array",\
                                           base_1 = b1, base_2 = b2)
                data[key] = np.abs(eps_stack[0, :]) + np.abs(eps_stack[1, :])
                lbl[key] = "$|\epsilon_{11}|+|\epsilon_{22}|$"

            elif key.lower() == "rotation":
                data[key] = self.ang[idx]
                lbl[key] = "Init Rot, (Deg)"

            elif key.lower() == "a_1":
                data[key] = self.getCellLengths(idx = idx, cell = 1)[:, 0]
                lbl[key] = "L $a_1$, ($\AA$)"

            elif key.lower() == "a_2":
                data[key] = self.getCellLengths(idx = idx, cell = 1)[:, 1]
                lbl[key] = "L $a_2$, ($\AA$)"

            elif key.lower() == "area":
                data[key] = self.getAreas(idx = idx, cell = 1)
                lbl[key] = "Area, ($\AA^2$)"

            elif key.lower() == "density":
                area = self.getAreas(idx = idx, cell = 1)
                base_area_1 = np.abs(np.cross(self.base_1[0, :2], self.base_1[1, :2]))
                base_area_2 = np.abs(np.cross(self.base_2[0, :2], self.base_2[1, :2]))

                vol = self.base_2[2, 2] * area
                atoms = self.atoms[idx] - area * (self.pos_1.shape[0] / base_area_1)
                norm_density = self.pos_2.shape[0] / (base_area_2 * self.base_2[2, 2])

                data[key] = atoms / (vol * norm_density)
                lbl[key] = "$\\rho$, (Atoms/$\AA^2$)"

            elif key.lower() == "e_int_c":
                if len(translation) > self.e_int_c.shape[1]:
                    string = "Translation (%i) outside e_int_c range (0,%i)"\
                             % (np.max(translation), self.e_int_c.shape[1])
                    ut.infoPrint(string)
                    return

                for t in translation:
                    iKey = "%s_%i" % (key, t)
                    data[iKey] = self.e_int_c[idx, :][:, t]
                    lbl[iKey] = "IE${_%i}^{C}$, ($eV/\AA2$)" % t

            elif key.lower() == "e_int_d":
                if len(translation) > self.e_int_d.shape[1]:
                    string = "Translation (%i) outside e_int_d range (0,%i)"\
                             % (np.max(translation), self.e_int_d.shape[1])
                    ut.infoPrint(string)
                    return

                for t in translation:
                    iKey = "%s_%i" % (key, t)
                    data[iKey] = self.e_int_d[idx, :][:, t]
                    lbl[iKey] = "IE${_%i}^{D}$, ($eV/\AA2$)" % t

            elif key.lower() == "e_int_diff_c":
                data[key] = np.max(self.e_int_c[idx, :], axis = 1) -\
                            np.min(self.e_int_c[idx, :], axis = 1)
                lbl[key] = "$\Delta$IE$^{C}$, ($eV/\AA^2$)"

            elif key.lower() == "e_int_diff_d":
                data[key] = np.max(self.e_int_d[idx, :], axis = 1) -\
                            np.min(self.e_int_d[idx, :], axis = 1)
                lbl[key] = "$\Delta$IE$^{D}$, ($eV/\AA^2$)"

            elif key.lower() == "w_sep_c":
                if len(translation) > self.w_sep_c.shape[1]:
                    string = "Translation (%i) outside w_sep_c range (0,%i)"\
                             % (np.max(translation), self.w_sep_c.shape[1])
                    ut.infoPrint(string)
                    return
                
                for t in translation:
                    iKey = "%s_%i" % (key, t)
                    data[iKey] = self.w_sep_c[idx, :][:, t]
                    lbl[iKey] = "W$_{%i}^{C}$, ($eV/\AA^2$)" % t

            elif key.lower() == "w_sep_d":
                if len(translation) > self.w_sep_d.shape[1]:
                    string = "Translation (%i) outside w_sep_d range (0,%i)"\
                             % (np.max(translation), self.w_sep_d.shape[1])
                    ut.infoPrint(string)
                    return
                
                for t in translation:
                    iKey = "%s_%i" % (key, t)
                    data[iKey] = self.w_sep_d[idx, :][:, t]
                    lbl[iKey] = "W$^D_{%i}$, ($eV/\AA^2$)" % t

            elif key.lower() == "w_sep_diff_c":
                data[key] = np.max(self.w_sep_c[idx, :], axis = 1) -\
                            np.min(self.w_sep_c[idx, :], axis = 1)
                lbl[key] = "$\Delta$W$^{C}$, ($eV/\AA^2$)"

            elif key.lower() == "w_sep_diff_d":
                data[key] = np.max(self.w_sep_d[idx, :], axis = 1) -\
                            np.min(self.w_sep_d[idx, :], axis = 1)
                lbl[key] = "$\Delta$W$^{D}$, ($eV/\AA^2$)"

            elif key.lower() == "w_seps_diff_c":
                data[key] = np.max(self.w_seps_c[idx, :], axis = 1) -\
                            np.min(self.w_seps_c[idx, :], axis = 1)
                lbl[key] = "$\Delta$W$^{SC}$, ($eV/\AA^2$)"

            elif key.lower() == "w_seps_diff_c":
                data[key] = np.max(self.w_seps_d[idx, :], axis = 1) -\
                            np.min(self.w_seps_d[idx, :], axis = 1)
                lbl[key] = "$\Delta$W$^{SD}$, ($eV/\AA^2$)"

            elif key.lower() == "w_seps_c":
                if len(translation) > self.w_seps_c.shape[1]:
                    string = "Translation (%i) outside w_seps_c range (0,%i)"\
                             % (np.max(translation), self.w_seps_c.shape[1])
                    ut.infoPrint(string)
                    return
                
                for t in translation:
                    iKey = "%s_%i" % (key, t)
                    data[iKey] = self.w_seps_c[idx, :][:, t]
                    lbl[iKey] = "W$^{SC}_{%i}$, ($eV/\AA^2$)" % t

            elif key.lower() == "w_seps_d":
                if len(translation) > self.w_seps_d.shape[1]:
                    string = "Translation (%i) outside w_seps_d range (0,%i)"\
                             % (np.max(translation), self.w_seps_d.shape[1])
                    ut.infoPrint(string)
                    return
                
                for t in translation:
                    iKey = "%s_%i" % (key, t)
                    data[iKey] = self.w_seps_d[idx, :][:, t]
                    lbl[iKey] = "W$^{SD}_{%i}$, ($eV/\AA^2$)" % t

            elif key.lower() == "other":
                data[key] = other
                lbl[key] = "Custom"

            else:
                string = "Unrecognized key: %s" % key
                ut.infoPrint(string)
                return

            
        return data, lbl



    def getCC(self, var, idx = None, translation = None, other = None,\
              verbose = 1, version = "pearson"):
        """Function for calculating correlation coefficients between specified variables
        
        For available values for "var" see function plotCorCoeff

        """

        if idx is None: idx = np.arange(self.atoms.shape[0])
        if translation is None: translation = [0]
        if isinstance(translation, (int, np.integer)): translation = [translation]
                
        data, lbl = self.getData(var = var, idx = idx, translation = translation)

        string = ""
        for i, key in enumerate(data):
            if i == 0:
                values = np.zeros((len(data), np.shape(data[key])[0]))

            if np.ndim(data[key]) == 1:
                values[i, :] = data[key]
            else:
                values[i, :] = data[key][:, 0]

        if "pearson".startswith(version.lower()):
            ccoef = np.zeros((values.shape[0], values.shape[0]))
            rho = np.zeros((values.shape[0], values.shape[0]))

            for i in range(0, values.shape[0]):
                for j in range(i + 1, values.shape[0]):
                    ccoef[i, j], rho[i, j] = stats.pearsonr(values[i, :], values[j, :])
                    ccoef[j, i] = ccoef[i, j]
                    rho[j, i] = rho[i, j]

            ccoef += np.identity(values.shape[0])

        elif "spearman".startswith(version.lower()):
            ccoef, rho = stats.spearmanr(values, axis = 1)

        if verbose > 0:
            head = "Cor-Coef for the following variables:"
            ut.infoPrint(head, sep_after = False)
            print("-" * len(head))
            for i, string in enumerate(var):
                ut.infoPrint("%s" % string, sep_before = False, sep_after = False)
            print("=" * len(head))

        return ccoef, rho, lbl



    def plotCorrCoef(self, var, idx = None, translation = None, norm = True, other = None,\
                         verbose = 1, cmap = "bwr", save = False, dpi = 100, format = "pdf",\
                         vmin = -1, vmax = 1, version = "pearson", rho = False, round = 2):
        """Function to plot the covariance matrix or the supplied variables

        Available values for "var" are
        ------------------------
        idx        = Index of current sorting
        eps_11     = Eps_11
        eps_22     = Eps_22
        eps_12     = Eps_12
        eps_mas    = Eps_mas
        eps_max    = max(eps_11, eps_22, eps_12)
        atoms      = Nr of atoms
        angle      = Angle between interface cell vectors
        rotation   = Initial rotation at creation
        norm       = Sqrt(eps_11^2+eps_22^2+eps_12^2)
        trace      = |eps_11|+|eps_22|
        norm_trace = Sqrt(eps_11^2+eps_22^2)
        a_1        = Length of interface cell vector a_1
        a_2        = Length of interface cell vector a_2
        area       = Area of the interface
        other      = Plot a custom array of values specified with keyword other. Length must match idx.
        e_int_c       = Interfacial energy, for specified translation(s)
        e_int_d       = Interfacial energy (DFT), for specified translation(s)
        e_int_diff_c  = Difference in Interfacial energy between translations
        e_int_diff_d  = Difference in Interfacial energy (DFT) between translations
        w_sep_c       = Work of separation, for specified translation(s)
        w_sep_d       = Work of separation (DFT), for specified translation(s)
        w_seps_c      = Work of separation (strained ref), for specified translation(s)
        w_seps_d      = Work of separation (strained ref) (DFT), for specified translation(s)
        w_sep_diff_c  = Difference in w_sep_c between tranlsations
        w_sep_diff_d  = Difference in w_sep_d (DFT) between tranlsations
        w_seps_diff_c = Difference in w_seps_c between translations
        w_seps_diff_d = Difference in w_seps_d (DFT) between translations
        """ 

        if "both".startswith(version.lower()):
            row = 1; col = 2
            ver = ["Pearson", "Spearman"]
            hFig = plt.figure(figsize = (11, 4.75))
        elif "pearson".startswith(version.lower()):
            row = 1; col = 1
            ver = ["Pearson"]
            hFig = plt.figure(figsize = (8, 6.5))
        elif "spearman".startswith(version.lower()):
            row = 1; col = 1
            ver = ["Spearman"]
            hFig = plt.figure(figsize = (8, 6.5))

        v = 0
        for i in range(row * col):
            if i > 0: v = 1

            ccoef, rho, lbl = self.getCC(var = var, idx = idx, other = other, version = ver[i],\
                                translation = translation, verbose = verbose - v)

            val = ccoef
            lbl = list(lbl.values())
            hAx = plt.subplot(row, col, i + 1)
            hIm = hAx.imshow(val, cmap = cmap)
            hIm.set_clim(vmin, vmax)

            hAx.set_xticks(np.arange(len(lbl)))
            hAx.set_yticks(np.arange(len(lbl)))

            hAx.set_xticklabels(lbl)
            hAx.set_yticklabels(lbl)

            plt.setp(hAx.get_xticklabels(), rotation = 45, ha = "right",
                     rotation_mode = "anchor")

            if (col * row == 1) or ((i + 1) == col * row):
                cbar = hAx.figure.colorbar(hIm, ax = hAx)

            for j in range(len(lbl)):
                for k in range(len(lbl)):
                    text = hAx.text(k, j, np.round(val[j, k], round), ha = "center",\
                                    va = "center", color = "k", fontsize = "x-small")

            hAx.set_title("%s Cor Coef" % ver[i])

        hFig.tight_layout()

        if save:
            if save is True:
                ut.save_fig(filename = "CorrCoefPlot.%s" % format, format = format,\
                         dpi = dpi, verbose = verbose)
            else:
                ut.save_fig(filename = save, format = format, dpi = dpi,\
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
            string = "Data saved to: %s" % filename
            ut.infoPrint(string)



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

        for i in range(self.e_int_c.shape[1]):
            key = "E_int_c_T%i" % (i)
            dataDict[key] = np.round(self.e_int_c[idx, i], prec)

        for i in range(self.w_sep_c.shape[1]):
            key = "W_sep_c_T%i" % (i)
            dataDict[key] = np.round(self.w_sep_c[idx, i], prec)

        for i in range(self.w_seps_c.shape[1]):
            key = "W_seps_c_T%i" % (i)
            dataDict[key] = np.round(self.w_seps_c[idx, i], prec)

        for i in range(self.e_int_d.shape[1]):
            key = "E_int_d_T%i" % (i)
            dataDict[key] = np.round(self.e_int_d[idx, i], prec)

        for i in range(self.w_sep_d.shape[1]):
            key = "W_sep_d_T%i" % (i)
            dataDict[key] = np.round(self.w_sep_d[idx, i], prec)

        for i in range(self.w_seps_d.shape[1]):
            key = "W_seps_d_T%i" % (i)
            dataDict[key] = np.round(self.w_seps_d[idx, i], prec)


        data = pd.DataFrame(dataDict)
        data.to_excel(filename)

        if verbose > 0:
            string = "Data written to Excel file: %s" % filename
            ut.infoPrint(string)



    def buildInterface(self, idx = 0, z_1 = 1, z_2 = 1, d = 2.5,\
                       verbose = 1, vacuum = 0, translation = None,\
                       surface = None, anchor = "@", alt_base = None):
        """Function for build a specific interface"""

        if verbose > 0:
            self.printInterfaces(idx = idx, anchor = anchor)

        """Get the distance between the top atom and the top of the cell"""
        void = self.base_1[2, 2] - np.max(self.pos_1[:, 2])
        d -= void

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
        rep_1 = np.zeros((3, 4))
        rep_1[0:2, 0:2] = self.rep_1[idx, :, :]
        rep_1[:, 2] = np.sum(rep_1, axis = 1)

        """Set all hi-lo limits for the cell repetitions"""
        h = 2
        rep_1 = [rep_1[0, :].min() - h, rep_1[0, :].max() + h,\
                 rep_1[1, :].min() - h, rep_1[1, :].max() + h,\
                 0, z_1 - 1]

        """Extend the cell as spcefied"""
        pos_1_ext, spec_1, mass_1 = ut.extendCell(base = self.base_1, rep = rep_1,\
                                                  pos = self.pos_1.T, spec = self.spec_1,\
                                                  mass = self.mass_1)

        """Working on interface B"""    
        """Set up the top interface with the correct repetitions and rotation"""
        rep_2 = np.zeros((3, 4))
        rep_2[0:2, 0:2] = self.rep_2[idx, :, :]
        rep_2[:, 2] = np.sum(rep_2, axis = 1)

        """Set all hi-lo limits for the cell repetitions"""
        h = 2
        rep_2 = [rep_2[0, :].min() - h, rep_2[0, :].max() + h,\
                 rep_2[1, :].min() - h, rep_2[1, :].max() + h,\
                 0, z_2 - 1]

        """Extend the cell as specified"""
        pos_2_ext, spec_2, mass_2 = ut.extendCell(base = self.base_2, rep = rep_2,\
                                                  pos = self.pos_2.T, spec = self.spec_2,\
                                                  mass = self.mass_2)

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
        if translation is not None:
            T = ut.getTranslation(translation, surface, verbose = verbose)[0]
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
        mass = np.concatenate((mass_1, mass_2))[keep]

        if alt_base is not None:
            """If supplied then build the interface using a different base for 
               the bottom cell. Intended to be used when switching from i.e. 
               LAMMPS to VASP and one wants to retain a fully relaxed bottom base."""
            if isinstance(alt_base, (int, np.integer)):
                alt_base = self.alt_base[alt_base, :2, :2]

            F[:2, :2] = np.matmul(alt_base, self.rep_1[idx, :, :])

        """Return to cartesian coordinates and change shape to (...,3)"""
        pos = np.matmul(F, pos_d).T

        """Add vacuum if specified"""
        F[2, 2] = F[2, 2] + vacuum
        if verbose: 
            string = "Z-distance fixed (between,above): %.2f | Vacuum added (above): %.2f"\
                     % (d + void, vacuum)
            ut.infoPrint(string)

        return F, pos, species, mass



    def exportInterface(self, idx = 0, z_1 = 1, z_2 = 1, d = 2.5,\
                        verbose = 1, format = "lammps",\
                        filename = None, vacuum = 0, translation = None,\
                        surface = None, anchor = "@", alt_base = None,\
                        kpoints = None):
        """Function for writing an interface to a specific file format"""

        if filename is None:
            if translation is None:
                filename = "interface_%s.%s" % (idx, format)
            else:
                filename = "interface_%s_T%s.%s" % (idx, translation, format)

        """Build the selected interface"""
        base, pos, type_n, mass = self.buildInterface(idx = idx, z_1 = z_1, z_2 = z_2, d = d,\
                                                verbose = verbose, vacuum = vacuum,\
                                                translation = translation, surface = surface,\
                                                anchor = anchor, alt_base = alt_base)

        """Sort first based on type then Z-position then Y-position"""
        ls = np.lexsort((pos[:, 1], pos[:, 2], type_n))

        """Sort all entries the same way"""
        type_n = type_n[ls]
        mass = mass[ls]
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

        """If specified write KPOINTS file"""
        if kpoints is not None:
            file_io.writeKPTS(cell = base, verbose = verbose, **kpoints)


    def buildSurface(self, idx = 0, surface = 1, z = 1, verbose = 1,\
                     strained = False, vacuum = 0, alt_base = None):
        """Function for build a specific interface"""
        
        if verbose > 0:
            self.printInterfaces(idx = idx)
            string = "Building surface nr %i in the interface shown above" % surface
            ut.infoPrint(string)

        """Basis for the selected surface, and the cell repetitions"""
        rep = np.zeros((3, 4))
        new_base = np.zeros((3, 3))
        if surface == 1:
            old_base = self.base_1
            spec = self.spec_1
            mass = self.mass_1
            pos = self.pos_1

            new_base[2, 2] = self.base_1[2, 2] * z
            new_base[0:2, 0:2] = self.cell_1[idx, :, :]
            
            rep[0:2, 0:2] = self.rep_1[idx, :, :]
            rep[:, 2] = np.sum(rep, axis = 1)
        elif surface == 2:
            old_base = self.base_2
            spec = self.spec_2
            mass = self.mass_2
            pos = self.pos_2

            new_base[2, 2] = self.base_2[2, 2] * z
            new_base[0:2, 0:2] = self.cell_2[idx, :, :]

            rep[0:2, 0:2] = self.rep_2[idx, :, :]
            rep[:, 2] = np.sum(rep, axis = 1)

        """Set all hi-lo limits for the cell repetitions"""
        h = 2
        rep = [rep[0, :].min() - h, rep[0, :].max() + h,\
               rep[1, :].min() - h, rep[1, :].max() + h,\
               0, z - 1]

        """Extend the cell as spcefied"""
        pos_ext, spec_ext, mass_ext = ut.extendCell(base = old_base, rep = rep,\
                                                    pos = pos.T, spec = spec,\
                                                    mass = mass)

        if surface == 1:
            if alt_base is not None:

                if isinstance(alt_base, (int, np.integer)):
                    alt_base = self.alt_base[alt_base, :2, :2]

                """Convert to direct coordinates"""
                pos_d = np.matmul(np.linalg.inv(new_base), pos_ext)

                """Update the base to the alternative one"""
                new_base[0:2, 0:2] = np.matmul(alt_base, self.rep_1[idx, :, :])

                """Convert positions back to cartesian coordinates"""
                pos_ext = np.matmul(new_base, pos_d)

                if verbose > 0:
                    string = "Surface 1 constructed with alternative base"
                    ut.infoPrint(string)

        elif surface == 2:
            """Initial rotation"""
            initRot = np.deg2rad(self.ang[idx])

            """Rotate the positions pos_rot = R*pos"""
            pos_ext = ut.rotate(pos_ext, initRot, verbose = verbose - 1)

            if strained or alt_base is not None:

                if isinstance(alt_base, (int, np.integer)):
                    alt_base = self.alt_base[alt_base, :2, :2]

                """If the cell is to be strained then convert to direct coordinates"""
                pos_d = np.matmul(np.linalg.inv(new_base), pos_ext)

                """Convert back to cartesian coordinates. Redefine the "new_base" to
                   be the strained base, with original/alternative base"""
                if alt_base is None:
                    new_base[0:2, 0:2] = self.cell_1[idx, :, :].copy()
                    string = "Surface 2 strained to match surface 1"
                    
                else:
                    new_base[0:2, 0:2] = np.matmul(alt_base, self.rep_1[idx, :, :])
                    string = "Surface 2 strained to match surface 1 with an alternative base"

                pos_ext = np.matmul(new_base, pos_d)

                if verbose > 0:
                    ut.infoPrint(string)

        """Convert the entire new cell to direct coordinates"""
        pos_d = np.matmul(np.linalg.inv(new_base), pos_ext)

        """Remove all positions outside [0, 1)"""
        pos_d = np.round(pos_d, 8)
        new_base = np.round(new_base, 8)

        keep = np.all(pos_d < 1, axis = 0) * np.all(pos_d >= 0, axis = 0)
        pos_d = pos_d[:, keep]
        species = spec_ext[keep]
        mass = mass_ext[keep]

        """Return to cartesian coordinates and change shape to (...,3)"""
        pos = np.matmul(new_base, pos_d).T

        """Add vacuum if specified"""
        new_base[2, 2] = new_base[2, 2] + vacuum
        if verbose: 
            string = "Vacuum added: %.2f"\
                     % (vacuum)
            ut.infoPrint(string)

        return new_base, pos, species, mass



    def exportSurface(self, idx = 0, z = 1, verbose = 1, format = "lammps",\
                      filename = None, vacuum = 0, surface = 1, strained = False,\
                      alt_base = None, kpoints = None):
        """Function for writing an interface to a specific file format"""

        if filename is None:
            filename = "surface_%s_S%s.%s" % (idx, surface, format)

        """Build the selected interface"""
        base, pos, type_n, mass = self.buildSurface(idx = idx, z = z, verbose = verbose,\
                                              vacuum = vacuum, surface = surface,\
                                              strained = strained, alt_base = alt_base)

        """Sort first based on type then Z-position then Y-position"""
        ls = np.lexsort((pos[:, 1], pos[:, 2], type_n))
        type_n = type_n[ls]
        mass = mass[ls]
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

        """If specified write KPOINTS file"""
        if kpoints is not None:
            file_io.writeKPTS(cell = base, verbose = verbose, **kpoints)



    def buildInterfaceStructure(self, idx = 0, z_1 = 1, z_2 = 1, d = 2.5,\
                                verbose = 1, filename = None, alt_base = None,\
                                vacuum = 0, translation = None, surface = None):
        """Function for writing an interface to a specific file format"""

        if filename is None:
            if translation is None:
                filename = "InterfaceStructure_%s" % (idx)
            else:
                filename = "InterfaceStructure_%s_T%s" % (idx, translation)

        """Build the selected interface"""
        base, pos, type_n, mass = self.buildInterface(idx = idx, z_1 = z_1, z_2 = z_2, d = d,\
                                                verbose = verbose, vacuum = vacuum, alt_base = alt_base,\
                                                translation = translation, surface = surface)

        """Sort first based on type then Z-position then Y-position"""
        ls = np.lexsort((pos[:, 1], pos[:, 2], type_n))
        type_n = type_n[ls]
        mass = mass[ls]
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



    def buildSurfaceStructure(self, idx = 0, z = 1, verbose = 1, alt_base = None,\
                              filename = None, vacuum = 0, surface = None):
        """Function for writing an interface to a specific file format"""

        if filename is None:
            filename = "SurfaceStructure_%s_S%s" % (idx, surface)

        """Build the selected interface"""
        base, pos, type_n, mass = self.buildSurface(idx = idx, z = z, verbose = verbose,\
                                              vacuum = vacuum, surface = surface, alt_base = alt_base)

        """Sort first based on type then Z-position then Y-position"""
        ls = np.lexsort((pos[:, 1], pos[:, 2], type_n))
        type_n = type_n[ls]
        mass = mass[ls]
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
