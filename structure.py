#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from interfaceBuilder import interface
from interfaceBuilder import file_io
from interfaceBuilder import inputs
from interfaceBuilder import utils as ut


class Structure():
    """
    Class for holding structure data from a general simulation.
    Single time snapshot.
    """

    __slots__ = ['cell', 'pos', 'type_n', 'type_i', 'idx',\
                 'mass', 'frozen', 'filename', 'pos_type']

    def __init__(self, 
                 cell = None,\
                 pos = None,\
                 type_n = None,\
                 type_i = None,\
                 idx = None,\
                 mass = None,\
                 frozen = None,\
                 filename = None,\
                 pos_type = None,\
                 load_from_file = None,\
                 load_from_input = None,\
                 format = None):        
        


        if load_from_file is not None:
            cell, pos, type, idx, mass = file_io.readData(load_from_file, format)
            if isinstance(type[0], (np.integer, int)):
                type_i = type
            else:
                type_n = type

            if np.all(pos >= 0) and np.all(pos <= 1):
                pos_type = "d"
            else:
                pos_type = "c"

            if filename is None: filename = load_from_file
     
        elif load_from_input is not None:
            cell, pos, type_n, mass = inputs.getInputs(lattice = load_from_input)

            if np.all(pos >= 0) and np.all(pos <= 1):
                pos_type = "d"
            else:
                pos_type = "c"

            type_i = np.zeros(type_n.shape[0])
            for i, item in enumerate(np.unique(type_n)):
                type_i[type_n == item] = i + 1

        """Simply placeholders for element names"""
        elements = [ "A", "B", "C", "D", "E", "F", "G", "H", "I",\
                     "J", "K", "L", "M", "N", "O", "P", "Q", "R"]
        if type_n is None:
            if type_i is None:
                type_i = np.ones(pos.shape[0])
                type_n = np.chararray(pos.shape[0], itemsize = 2)
                type_n[:] = "A"
            else:
                type_n = np.chararray(pos.shape[0], itemsize = 2)
                for i, item in enumerate(np.unique(type_i)):
                    type_n[type_i == item] = elements[i]
        else:
            if type_i is None:
                type_i = np.ones(type_n.shape[0])
                for i, item in enumerate(np.unique(type_n)):
                    type_i[type_n == item] = i + 1

        if idx is None: idx = np.arange(pos.shape[0])
        if mass is None: mass = type_i
        if frozen is None: frozen = np.zeros((pos.shape[0], 3), dtype = bool)
        if pos_type is None: pos_type = "c"
        if filename is None: filename = "structure_obj"

        self.filename = filename
        self.cell = cell
        self.pos = pos
        self.type_n = type_n
        self.type_i = type_i
        self.pos_type = pos_type.lower()
        self.frozen = frozen

        """Assign idx if dimensions match otherwise assign a range"""
        if idx.shape[0] != self.pos.shape[0]:
            self.idx = np.arange(self.pos.shape[0])
        else:
            self.idx = idx

        mass = np.array(mass)
        if mass.shape[0] == self.pos.shape[0]:
            """If shape of mass == nr o atoms simply assign it"""
            self.mass = mass

        elif mass.shape[0] == np.unique(self.type_i).shape[0]:
            """If shape mass == unique types, assign to types in order"""
            self.mass = np.ones(self.pos.shape[0])
            for i, item in enumerate(np.unique(self.type_i)):
                self.mass[self.type_i == item] = mass[i]
                    
        else:
            """Else simply assign the type_i value as placeholder"""
            self.mass = type_i

        self.sortStructure(verbose = 0)


    def sortStructure(self, sort = "type", reset_idx = False, verbose = 1):
        """Function for sorting the structure

        sort = str("type"/"index"), Sort by type-z-y-x or sort by index as loaded
        from simulation file.

        reset_idx = bool, If True then reset the index of the structure

        verbose = int, Print extra information
        """

        if sort.lower() == "type":
            """Sorts the structure based on type_i then z then y then x"""
            si = np.lexsort((self.pos[:, 0], self.pos[:, 1], self.pos[:, 2], self.type_i))

        elif sort.lower() == "index":
            """Sort by the idx property, (index as written by the simulation programs)"""
            si = np.argsort(self.idx)

        if verbose > 0:
            string = "Sorting structure by %s" % sort.lower()
            ut.infoPrint(string)

        """Sort as specified"""
        self.pos = self.pos[si]
        self.type_i = self.type_i[si]
        self.type_n = self.type_n[si]
        self.frozen = self.frozen[si]
        self.mass = self.mass[si]

        if reset_idx:
            self.resetIndex(verbose = verbose)
        else:
            self.idx = self.idx[si]


    def combineTypes(self, combine, element, verbose = 1):
        """Function for combining atom types in to the same group.

           combine = list, index of groups to combine

           element = str, element for the new combined group
        """

        name = np.chararray(1, itemsize = 2)
        name[0] = element

        mask = np.zeros(self.type_i.shape[0], dtype = bool)
        for i in combine:
            mask[self.type_i == i] = True

        negative = np.logical_not(mask)
        if np.any(self.type_n[negative] == name):
            string = "Can not change new group element to %s as that already exist in another group"
            ut.infoPrint(string)
            return

        self.type_i[mask] = np.min(combine)
        self.type_n[mask] = element

        last = 1
        for i, item in enumerate(np.unique(self.type_i)):
            self.type_i[self.type_i == item] = i + 1

        if verbose > 0:
            string = "Combining types %s into type %s with element type %s"\
                     % (combine, np.min(combine), element)
            ut.infoPrint(string)


    def resetIndex(self, verbose = 1):
        """Function for reseting the atomic indicies

        verbose = int, Print extra information
        """

        self.idx = np.arange(self.pos.shape[0])
        if verbose > 0:
            string = "Reseting atom index"
            ut.infoPrint(string)


    def printStructure(self):
        """Function to print formated output of the structure"""

        string = "%s" % self.filename
        print("\n%s" % string)
        print("-" * len(string))
        
        print("Cell")
        print("-" * 32)
        for i in range(self.cell.shape[0]):
            print("%10.4f %10.4f %10.4f" % (self.cell[i, 0], self.cell[i, 1], self.cell[i, 2]))
        print("-" * 32)

        string = "%5s %5s %5s %12s %12s %12s" %\
                ("Index", "Name", "Type", "Pos x    ", "Pos y    ", "Pos z    ")
        print(string)
        print("-" * len(string))

        for i in range(self.pos.shape[0]):
            print("%5i %5s %5i %12.5f %12.5f %12.5f" % (self.idx[i], self.type_n[i].decode("utf-8"),\
                                                        self.type_i[i], self.pos[i, 0], self.pos[i, 1],\
                                                        self.pos[i, 2]))

        print("-" * len(string))
        string = "Nr of Atoms: %i | Nr of Elements: %i" % (self.pos.shape[0], np.unique(self.type_i).shape[0])
        print(string)



    def alignStructure(self, dim = [1, 0, 0], align = [1, 0, 0], verbose = 1):
        """Function for aligning a component of the structure in a specific dimension

        dim = [float, float, float], the cell will be aligned to have this
        cell vector entierly in the direction of the cartesian axis supplied
        in the align parameter. Dim in direct coordinates.
                          
        align = [float, float, float], cartesian axis to align dim to. Align in cartesian coordinates.

        verbose = int, Print extra information
        """

        if align[2] != 0:
            align[2] = 0
            print("Only alignment of cells with the z-axis ortogonal to the xy-plane is supported")
            print("The Z-component is discarded")
        
        dim = np.array(dim)
        align = np.array(align)

        """Get the cartesian direction of the dim which is to be aligned"""
        dim_cart = np.matmul(self.cell, dim)

        if dim_cart[2] != 0:
            dim_cart[2] = 0
            print("Specified dimension has cartesian components in the Z direction.")
            print("The rotation is made ortogonal to the xy-plane.")
        
        """Get the angle between dim and align"""
        aRad = np.arccos(np.dot(dim_cart, align) / (np.linalg.norm(dim_cart) * np.linalg.norm(align)))

        """Check in which direction the rotation should be made"""
        dir = np.cross(dim_cart, align)[2]
        if dir < 0:
            aRad = -aRad
        
        R = np.array([[np.cos(aRad), -np.sin(aRad), 0],
                      [np.sin(aRad),  np.cos(aRad), 0],
                      [           0,             0, 1]]) 

        self.cell = np.matmul(R, self.cell)
        self.pos = np.matmul(R, self.pos.T).T


    def getBoxLengths(self):
        """Function for getting the box side lengths in orthogonal x,y,z"""

        return self.cell[np.identity(3, dtype = bool)]


    def getAtoms(self, mode):
        """get index of atoms that match supplied criteria

        Mode is supplied as a dict with keyword specified as below
        and containing a list of 

        i.e. {"box": [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi], "type": [2, 3]}
        Slice out box within coordinates x,y,z and additionally of type 2 or 3

        Mode and options
        ----------------
        box    - [x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]
        sphere - [x, y, z, radius]
        type   - [list of all types to include]
        idx    - [list of all atomic indices to include]
        """

        print("Not defined")
        

    def getNeighborDistance(self, idx = None, r = 6, idx_to = None,\
                            extend = np.array([1, 1, 1], dtype = bool),\
                            verbose = 1):
        """Function for getting the distance between specified atoms within
        radius r

        idx = int, [int,], Index from which to calculate nearest neighbors

        r = float, Radius or NN calculation

        idx_to = int, [int,], Calculate the NN considering only these atoms

        extend = np.ndarray([1/0, 1/0, 1/0]), Extend the cell if needed in
        specified x, y, z directions

        verbose = int, Print extra information
        """

        """Check some defaults"""
        if idx is None: idx = np.arange(self.pos.shape[0])
        if isinstance(idx, (int, np.integer)): idx = np.array([idx])
        if idx_to is None: idx_to = np.arange(self.pos.shape[0])
        if isinstance(idx_to, (int, np.integer)): idx_to = np.array([idx_to])
        extend = np.array(extend, dtype = bool)
        
        """Change to cartesian coordinates"""
        self.dir2car()

        """Check the rough maximum possible extent for relevant atoms"""
        max_pos = np.max(self.pos[idx, :], axis = 0) + r
        min_pos = np.min(self.pos[idx, :], axis = 0) - r
        lim = np.all(self.pos < max_pos, axis = 1) *\
              np.all(self.pos > min_pos, axis = 1)
        idx_to = np.intersect1d(self.idx[lim], idx_to)
        if verbose > 0:
            string = "Considering idx_to within [%.2f, %.2f] (x), [%.2f, "\
                     " %.2f] (y), [%.2f,  %.2f] (z)" % (min_pos[0], max_pos[0],\
                     min_pos[1], max_pos[1], min_pos[2], max_pos[2])
            ut.infoPrint(string)

        """Cell extension to comply with the sepecified r value"""
        box = self.getBoxLengths()
        rep = np.ceil(r / box) - 1
        rep = rep.astype(np.int)
        rep[np.logical_not(extend)] = 0

        """If the box < r then extend the box. Otherwise wrap the cell"""
        if np.any(box < r):

            if verbose > 0:
                string = "Replicating cell by %i, %i, %i (x, y, z)"\
                         % (rep[0] + 1, rep[1] + 1, rep[2] + 1)
                ut.infoPrint(string)

            pos_to, cell = self.getExtendedPositions(x = rep[0], y = rep[1], z = rep[2],\
                                                  idx = idx_to, return_cart = True,\
                                                  verbose = verbose - 1)

            """Change to cartesian coordinates"""
            self.dir2car()

            """Change back to direct coordinates using the new extended cell"""
            pos_to = np.matmul(np.linalg.inv(cell), pos_to.T).T
            pos_from = np.matmul(np.linalg.inv(cell), self.pos.T).T 
        else:
            """Change to direct coordinates"""
            self.car2dir()

            if verbose > 0:
                string = "Cell is only wrapped, not extended"
                ut.infoPrint(string)

            pos_to = self.pos.copy()[idx_to, :]
            pos_from = self.pos.copy()
            cell = self.cell

        ps = pos_to.shape[0]
        dist = np.zeros((np.shape(idx)[0] * ps, 3))

        """Measure distances between all specified atoms, wrap cell"""
        for i, item in enumerate(idx):
            d = pos_to - pos_from[[item], :]
        
            d[d >  0.5] -= 1
            d[d < -0.5] += 1

            dist[i * ps : (i + 1) * ps, :] = d

        """Convert to cartesian coordinates"""
        dist = np.matmul(cell, dist.T).T

        """Calculate the distances"""
        dist = np.linalg.norm(dist, axis = 1)

        """Remove distances outside of radius r"""
        dist = dist[dist < r]

        """Remove the 0 distances, (atom to it self)"""
        dist = dist[dist > 0]

        return dist



    def getNearestNeighbors(self, idx = None, idx_to = None, NN = 8,\
                            verbose = 1, limit = np.array([5, 5, 5]),\
                            extend = np.array([1, 1, 1], dtype = bool)):
        """Function for getting index and distance to nearest neighbors 
        of specified atoms

        idx = int, [int,], Index from which to calculate nearest neighbors

        NN = int, Number of nearest neighbors to keep

        idx_to = int, [int,], Calculate the NN considering only these atoms

        limit = np.ndarray([float, float, float]), Limit around the maximum extent
        of the included atoms, to speed up calculations

        verbose = int, Print extra information
        """

        """Check some defaults"""
        if idx is None: idx = np.arange(self.pos.shape[0])
        if isinstance(idx, (int, np.integer)): idx = np.array([idx])
        if idx_to is None: idx_to = np.arange(self.pos.shape[0])
        if isinstance(idx_to, (int, np.integer)): idx_to = np.array([idx_to])
        
        """Change to cartesian coordinates"""
        self.dir2car()

        """Do a rough check which atoms must be included in the NN search"""
        max_pos = np.max(self.pos[idx, :], axis = 0) + limit
        min_pos = np.min(self.pos[idx, :], axis = 0) - limit

        lim = np.all(self.pos < max_pos, axis = 1) *\
              np.all(self.pos > min_pos, axis = 1)
        idx_to = np.intersect1d(self.idx[lim], idx_to)

        if verbose > 0:
            string = "Considering idx_to within [%.2f, %.2f] (x), [%.2f, "\
                     " %.2f] (y), [%.2f,  %.2f] (z)" % (min_pos[0], max_pos[0],\
                     min_pos[1], max_pos[1], min_pos[2], max_pos[2])
            ut.infoPrint(string)

        """Cell extension to comply with the sepecified limit"""
        box = self.getBoxLengths()
        rep = np.ceil(limit / box) - 1
        rep = rep.astype(np.int)
        rep[np.logical_not(extend)] = 0

        if np.any(rep > 0):
            if verbose > 0:
                string = "Replicating cell by %i, %i, %i (x, y, z)"\
                         % (rep[0] + 1, rep[1] + 1, rep[2] + 1)
                ut.infoPrint(string)

            """Extend teh cell"""
            pos_to, cell = self.getExtendedPositions(x = rep[0], y = rep[1], z = rep[2],\
                                                     idx = idx_to, return_cart = True,\
                                                     verbose = verbose - 1)

            """Change to cartesian coordinates"""
            self.dir2car()

            """Change back to direct coordinates using the new extended cell"""
            pos_to = np.matmul(np.linalg.inv(cell), pos_to.T)
            pos_from = np.matmul(np.linalg.inv(cell), self.pos.T) 

        else:
            """Change to direct coordinates"""
            self.car2dir()

            if verbose > 0:
                string = "Cell is only wrapped, not extended"
                ut.infoPrint(string)

            pos_to = self.pos.copy()[idx_to, :].T
            pos_from = self.pos.copy().T
            cell = self.cell

        if pos_to.shape[1] - 1 < NN:
            string = "Within current limits (%.2f, %.2f, %.2f) fewer NN (%i) "\
                     "are present than specified (%i)" % (limit[0], limit[1],\
                     limit[2], pos_to.shape[1] - 1, NN)
            ut.infoPrint(string)
            return

        distance = np.zeros((np.shape(idx)[0], NN))

        """Measure distances between all specified atoms, wrap cell"""
        for i, item in enumerate(idx):
            d = pos_to - pos_from[:, [item]]

            d[d >  0.5] -= 1
            d[d < -0.5] += 1

            """Convert to cartesian coordinates"""
            c = np.matmul(cell, d)

            """Calculate distances"""
            dist = np.sqrt(c[0, :]**2 + c[1, :]**2 + c[2, :]**2)

            """Remove distance to the same atom"""
            mask = dist > 0
            dist = dist[mask]

            """Sort distances"""
            si = np.argsort(dist)

            """Pick out the NN nearest in all variables"""
            distance[i, :] = dist[si][:NN]

        return distance


    def getNearestNeighborCollection(self, idx = None, idx_to = None, NN = 8,\
                                     verbose = 1, limit = np.array([5, 5, 5]),\
                                     extend = np.array([1, 1, 1], dtype = bool)):
        """Function for getting nearest neighbors around specified atoms to 
        specified atoms collected as an average with a standard deviation

        idx = int, [int,], Index from which to calculate nearest neighbors

        NN = int, Number of nearest neighbors to keep

        idx_to = int, [int,], Calculate the NN considering only these atoms

        limit = np.ndarray([float, float, float]), Limit around the maximum extent
        of the included atoms, to speed up calculations

        extend = np.ndarray([1/0, 1/0, 1/0]), Extend the cell if needed in
        specified x, y, z directions

        verbose = int, Print extra information
        """

        """Check some defaults"""
        if idx is None: idx = np.arange(self.pos.shape[0])
        if isinstance(idx, (int, np.integer)): idx = np.array([idx])
        if idx_to is None: idx_to = np.arange(self.pos.shape[0])
        if isinstance(idx_to, (int, np.integer)): idx_to = np.array([idx_to])
        
        """Change to cartesian coordinates"""
        self.dir2car()

        distance = self.getNearestNeighbors(idx = idx, idx_to = idx_to, NN = NN,\
                                            verbose = verbose, limit = limit,\
                                            extend = extend)

        """Return if less than NN neighbors could be found"""
        if distance is None: 
            return

        if verbose > 0:
            string = "Calculated %i nearest neighbors for %i atoms" % (NN, distance.shape[0])
            ut.infoPrint(string)

        dist_mean = np.mean(distance, axis = 0)
        dist_std = np.std(distance, axis = 0)

        return dist_mean, dist_std



    def getNearestAngles(self, idx, idx_to, verbose = 1):
        """Function for getting the angles between specified atoms
           to specified atoms"""

        print("Get angles")



    def plotNNC(self, idx, idx_to = None, NN = 8, verbose = True,\
               handle = False, row = 1, col = 1, N = 1, save = False,\
               format = "pdf", dpi = 100, legend = None, **kwargs):
        """Function for plotting a nearest neighbor collection with std

        idx = int, [int,], Index from which to calculate nearest neighbors

        NN = int, Number of nearest neighbors to keep

        idx_to = int, [int,], Calculate the NN considering only these atoms

        handle = bool, If True only prepare the axis don't draw the plot

        row = int, Rows if used in subplots

        col = int, Columns if used in subplots

        N = int, Nr of plot if used in subplots

        save = bool or str, Name to save the file to or save to default name
        if True

        format = valid matplotlib format, Format to save the plot in

        dpi = int, DPI used when saving the plot

        legend = [str,], Legend for the different entries

        **kvargs = valid matplotlib errorbar kwargs
        """

        lbl_1 = None
        if idx is None:
            idx = [np.arange(self.pos.shape[0])]
        elif isinstance(idx, (int, np.integer)):
            idx = [np.array([idx])]
        elif isinstance(idx[0], (int, np.integer)):
            idx = [idx]
        elif isinstance(idx, str) and idx.lower() == "species":
            idx, lbl_1 = self.getElementIdx()[:2]

        lbl_2 = None
        if idx_to is None:
            idx_to = [np.arange(self.pos.shape[0])]
        elif isinstance(idx_to, (int, np.integer)):
            idx_to = [np.array([idx_to])]
        elif isinstance(idx_to[0], (int, np.integer)):
            idx_to = [idx_to]
        elif isinstance(idx_to, str) and idx_to.lower() == "species":
            idx_to, lbl_2 = self.getElementIdx()[:2]


        if len(idx) == 1:
            l_idx = np.zeros(len(idx_to), dtype = np.int)
        else:
            l_idx = np.arange(len(idx), dtype = np.int)
            
        if len(idx_to) == 1:
            l_idx_to = np.zeros(len(idx), dtype = np.int)
        else:
            l_idx_to = np.arange(len(idx_to), dtype = np.int)
        
        if l_idx.shape[0] != l_idx_to.shape[0]:
            string = "Length of idx and idx_to does not match (%i, %i). "\
                     "Can be (1,N), (N,1) or (N,N)"\
                     % (l_idx.shape[0], l_idx_to.shape[0])
            ut.infoPrint(string)
            return

        x = []; y = []; s = []
        for i in range(l_idx.shape[0]):
            d_mean, d_std = self.getNearestNeighborCollection(idx = idx[l_idx[i]],\
                                                              idx_to = idx_to[l_idx_to[i]],\
                                                              NN = NN, verbose = verbose)
            
            x.append(np.arange(1, d_mean.shape[0] + 1))
            y.append(d_mean)
            s.append(d_std)

        if not handle:
            hFig = plt.figure()

        """Set some defaults"""
        ls = kwargs.pop("linestyle", "--")
        lw = kwargs.pop("linewidth", 0.5)
        m = kwargs.pop("marker", "o")
        ms = kwargs.pop("markersize", 3)
        cs = kwargs.pop("capsize", 2)
        elw = kwargs.pop("elinewidth", 1)

        hAx = plt.subplot(row, col, N)
        label = "_ignore"
        for i, item in enumerate(y):
            if legend is not None:
                if legend.lower() == "idx":
                    label = "%i -> %i" % (l_idx[i], l_idx_to[i])
                else:
                    label = legend[i]
            elif lbl_1 is not None and lbl_2 is not None:
                label = "%2s -> %2s" % (lbl_1[i], lbl_2[i])
            elif lbl_1 is not None:
                label = "%2s -> %i" % (lbl_1[i], l_idx_to[i])
            elif lbl_2 is not None:
                label = "%i -> %2s" % (l_idx[i], lbl_2[i])

            hAx.errorbar(x[i], y[i], yerr = s[i], linestyle = ls, marker = m, capsize = cs,\
                         elinewidth = elw, markersize = ms, linewidth = lw, label = label, **kwargs)

        hAx.set_xlabel("Neighbor")
        hAx.set_ylabel("Distance, $(\AA)$")

        if label != "_ignore":
            hAx.legend(framealpha = 1, loc = "upper left")

        hAx.set_title("Nearest Neighbor Distances")
        plt.tight_layout()
        if save:
            if save is True:
                ut.save_fig(filename = "NNC.%s" % (format), format = format,\
                            dpi = dpi, verbose = verbose)
            else:
                ut.save_fig(filename = save, format = format, dpi = dpi,\
                            verbose = verbose)
            plt.close()
        else:
            plt.show()




    def plotNN(self, idx, idx_to = None, NN = 8, verbose = True,\
               handle = False, row = 1, col = 1, N = 1, save = False,\
               format = "pdf", dpi = 100, **kwargs):
        """Function to plot the distances to the N nearest neighbors

        idx = int, [int,], Index from which to calculate nearest neighbors

        NN = int, Number of nearest neighbors to keep

        idx_to = int, [int,], Calculate the NN considering only these atoms

        handle = bool, If True only prepare the axis don't draw the plot

        row = int, Rows if used in subplots

        col = int, Columns if used in subplots

        N = int, Nr of plot if used in subplots

        save = bool or str, Name to save the file to or save to default name
        if True

        format = valid matplotlib format, Format to save the plot in

        dpi = int, DPI used when saving the plot

        legend = [str,], Legend for the different entries

        **kvargs = valid matplotlib errorbar kwargs
        """

        if isinstance(idx, (np.integer, int)): idx = np.array([idx])
        if idx_to is None: idx_to = np.arange(self.pos.shape[0])
        
        distance = self.getNearestNeighbors(idx = idx, idx_to = idx_to,\
                                            NN = NN, verbose = verbose)[1]

        x = np.arange(1, distance.shape[1] + 1)
        
        if not handle:
            hFig = plt.figure()

        """Set some defaults"""
        ls = kwargs.pop("linestyle", "--")
        lw = kwargs.pop("linewidth", 0.5)
        m = kwargs.pop("marker", "o")
        ms = kwargs.pop("markersize", 3)

        hAx = plt.subplot(row, col, N)
        for i in range(np.shape(distance)[0]):
            hAx.plot(x, distance[i, :], linestyle = ls, marker = m, markersize = ms,\
                     linewidth = lw, **kwarg)
        

        hAx.set_title("Nearest Neighbor Distances")
        hAx.set_xlabel("Neighbor")
        hAx.set_ylabel("Distance, $(\AA)$")
        
        plt.tight_layout()
        if save:
            if save is True:
                ut.save_fig(filename = "NN.%s" % (format), format = format,\
                         dpi = dpi, verbose = verbose)
            else:
                ut.save_fig(filename = save, format = format, dpi = dpi,\
                         verbose = verbose)
            plt.close()
        else:
            plt.show()
            



    def getRDF(self, idx = None, idx_to = None, r = 6, dr = 0.1, bins = None,\
               extend = np.array([1, 1, 1], dtype = bool), edges = False,\
               verbose = 1):
        """Function for getting the radial distribution function around and 
        to specified atoms

        idx = int, [int,], Index from which to calculate nearest neighbors

        idx_to = int, [int,], Calculate the NN considering only these atoms

        r = float, Cut off used in the RDF

        dr = float, bins size used in the RDF

        bins = int, Alternative to dr, specify the total number of bins

        extend = np.ndarray([1/0, 1/0, 1/0]), Allow the cell to be extended in
        the specified x, y, z directions

        edges = bool, Include both edges

        verbose = int, Print extra information
        """

        """Check some defaults"""
        if idx is None: idx = np.arange(self.pos.shape[0])
        if isinstance(idx, (int, np.integer)): idx = np.array([idx])
        if idx_to is None: idx_to = np.arange(self.pos.shape[0])
        if isinstance(idx_to, (int, np.integer)): idx_to = np.array([idx_to])
        extend = np.array(extend, dtype = bool)

        dist = self.getNeighborDistance(idx = idx, idx_to = idx_to, r = r,\
                                        extend = extend, verbose = verbose)

        """If bins is specified it is used over dr"""
        if bins is None:
            bins = np.arange(0, r + dr, dr)

        cnt, bin = np.histogram(dist, bins = bins, range = (0, r))

        """Get volume of the radial spheres, V=4/3*pi*(bin[1]^3 - bin[0]^3) """
        V = 4/3*np.pi * (bin[1:]**3 - bin[:-1]**3)

        """Calculate the cumulative distribution without normalizing by V"""
        N = np.shape(idx)
        tot = np.cumsum(cnt / N)

        """Normalize the RD count by V and nr of atoms which the RDF is centered around"""
        cnt = cnt / (N * V)

        if not edges:
            bin = bin[:-1] + bin[1] / 2

        return cnt, bin, tot



    def plotRDF(self, idx = None, idx_to = None, r = 6, dr = 0.1, bins = None,\
                extend = np.array([1, 1, 1], dtype = bool), cumulative = False, legend = None,\
                row = 1, col = 1, N = 1, handle = False, save = False, format = "pdf",\
                dpi = 100, verbose = 1, **kwargs):
        """Function for ploting the RDF and cumulative distribution

        idx = int, [int,], Index from which to calculate nearest neighbors

        idx_to = int, [int,], Calculate the NN considering only these atoms

        r = float, Cut off used in the RDF

        dr = float, bins size used in the RDF

        bins = int, Alternative to dr, specify the total number of bins

        extend = np.ndarray([1/0, 1/0, 1/0]), Allow the cell to be extended in
        the specified x, y, z directions

        edges = bool, Include both edges

        cumulative = bool, Add the cumulative RDF value to a right y-axis

        legend = [str,] legend inputs

        handle = bool, If True only prepare the axis don't draw the plot

        row = int, Rows if used in subplots

        col = int, Columns if used in subplots

        N = int, Nr of plot if used in subplots

        save = bool or str, Name to save the file to or save to default name
        if True

        format = valid matplotlib format, Format to save the plot in

        dpi = int, DPI used when saving the plot

        kwargs = valid matplotlib plot kwargs
        """
        
        lbl_1 = None
        if idx is None:
            idx = [np.arange(self.pos.shape[0])]
        elif isinstance(idx, (int, np.integer)):
            idx = [np.array([idx])]
        elif isinstance(idx[0], (int, np.integer)):
            idx = [idx]
        elif isinstance(idx, str) and idx.lower() == "species":
            idx, lbl_1 = self.getElementIdx()[:2]

        lbl_2 = None
        if idx_to is None:
            idx_to = [np.arange(self.pos.shape[0])]
        elif isinstance(idx_to, (int, np.integer)):
            idx_to = [np.array([idx_to])]
        elif isinstance(idx_to[0], (int, np.integer)):
            idx_to = [idx_to]
        elif isinstance(idx_to, str) and idx_to.lower() == "species":
            idx_to, lbl_2 = self.getElementIdx()[:2]
            

        if len(idx) == 1:
            l_idx = np.zeros(len(idx_to), dtype = np.int)
        else:
            l_idx = np.arange(len(idx), dtype = np.int)
            
        if len(idx_to) == 1:
            l_idx_to = np.zeros(len(idx), dtype = np.int)
        else:
            l_idx_to = np.arange(len(idx_to), dtype = np.int)

        if l_idx.shape[0] != l_idx_to.shape[0]:
            string = "Length of idx and idx_to does not match (%i, %i). "\
                     "Can be (1,N), (N,1) or (N,N)"\
                     % (l_idx.shape[0], l_idx_to.shape[0])
            ut.infoPrint(string)
            return
            
        y = []; yt = []
        for i in range(l_idx.shape[0]):
            cnt, bin, tot = self.getRDF(idx = idx[l_idx[i]], idx_to = idx_to[l_idx_to[i]], r = r,\
                                        dr = 0.1, bins = bins, extend = extend,\
                                        edges = False, verbose = verbose)
            
            y.append(cnt)
            yt.append(tot)

        if not handle:
            hFig = plt.figure()

        hAx = plt.subplot(row, col, N)
        if cumulative:
            hAxR = hAx.twinx()
        label = "_ignore"
        for i, item in enumerate(y):
            if legend is not None:
                if legend.lower() == "idx":
                    label = "%i -> %i" % (l_idx[i], l_idx_to[i])
                else:
                    label = legend[i]
            elif lbl_1 is not None and lbl_2 is not None:
                label = "%2s -> %2s" % (lbl_1[i], lbl_2[i])
            elif lbl_1 is not None:
                label = "%2s -> %i" % (lbl_1[i], l_idx_to[i])
            elif lbl_2 is not None:
                label = "%i -> %2s" % (l_idx[i], lbl_2[i])

            hL = hAx.plot(bin, item, linestyle = "-", label = label, **kwargs)
            if cumulative:
                hAxR.plot(bin, yt[i], linestyle = "--", color = hL[-1].get_color(), **kwargs)

        hAx.set_xlabel("Radius, $\AA$")
        hAx.set_ylabel("Atoms / (Atom * Volume), ($1/\AA^3$)")
        if label != "_ignore":
            hAx.legend(framealpha = 1, loc = "upper left")

        hAx.set_title("RDF")
        plt.tight_layout()
        if save:
            if save is True:
                ut.save_fig(filename = "RDF.%s" % (format), format = format,\
                         dpi = dpi, verbose = verbose)
            else:
                ut.save_fig(filename = save, format = format, dpi = dpi,\
                         verbose = verbose)
            plt.close()
        else:
            plt.show()


    def getElementIdx(self):
        """Function for getting the atomic indices for each element"""

        index = np.arange(self.pos.shape[0])
        idx = []; element = []; nr = []
        for i in np.unique(self.type_n):
            idx.append(index[self.type_n == i])
            nr.append(self.type_i[self.type_n == i][0])
            element.append(i.decode("utf-8"))

        return idx, element, nr



    def extendStructure(self, x = 1, y = 1, z = 1, reset_index = False, verbose = 1):
        """Function for repeating the cell in x, y or z direction

        x = int, extend this many times

        y = int, extend this many times

        z = int, extend this many times

        reset_index = bool, Reset the index after extending the structure

        verbose = Print Extra information
        """

        """Change to direct coordinates"""
        self.car2dir()
        
        if x < 1: x = 1
        if y < 1: y = 1
        if z < 1: z = 1

        x = np.int(np.ceil(x))
        y = np.int(np.ceil(y))
        z = np.int(np.ceil(z))

        l = self.pos.shape[0]

        self.type_n = np.tile(self.type_n, x*y*z)
        self.type_i = np.tile(self.type_i, x*y*z)
        self.mass = np.tile(self.mass, x*y*z)
        self.frozen = np.tile(self.frozen, (x*y*z, 1))

        self.idx = np.tile(self.idx, x*y*z)
        self.pos = np.tile(self.pos, (x*y*z, 1))

        n = 0
        for i in range(z):
            for j in range(y):
                for k in range(x):
                    self.pos[l*n:l*(n+1), :] += (k, j, i)
                    self.idx[l*n:l*(n+1)] += l*n 
                    n += 1

        """Sort resulting properties, type-z-y-x"""
        s = np.lexsort((self.pos[:, 0], self.pos[:, 1],\
                        self.pos[:, 2], self.type_i))

        self.pos = self.pos[s, :]
        self.type_i = self.type_i[s]
        self.type_n = self.type_n[s]
        self.mass = self.mass[s]
        if reset_index:
            self.idx = np.arange(self.pos.shape[0])
        else:
            self.idx = self.idx[s]

        """Convert the cell back to cartesian coordinates"""
        self.dir2car()

        """Extend the cell"""
        self.cell[:, 0] *= x
        self.cell[:, 1] *= y
        self.cell[:, 2] *= z



    def addVacuum(self, va = 1, vb = 1, vc = 1, verbose = 1):
        """Function for adding a vacuum layer to a structure"""

        """Change to cartesian coordinates"""
        self.dir2car()

        """Set up matrix to extend the cell dimensions as specified"""
        vacuum = np.array([[va, 0, 0], [0, vb, 0], [0, 0, vc]])

        """Extend with vacuum"""
        self.cell = np.matmul(self.cell, vacuum)

        if verbose > 0:
            
            for i in range(3):
                string = "%10.4f %10.4f %10.4f" % (self.cell[i, 0], self.cell[i, 1], self.cell[i, 2])
                if i == 0:
                    ut.infoPrint(string, sep_before = True, sep_after = False)
                elif i == 1:
                    ut.infoPrint(string, sep_before = False, sep_after = False)
                else:
                    ut.infoPrint(string, sep_before = False, sep_after = True)



    def stretchCell(self, x = 1, y = 1, z = 1, transform = None, verbose = 1):
        """Function for stretching the cell and positions of a structure"""

        """Change to direct coordinates"""
        self.car2dir()

        """If the full transformation matrix is supplied it is used otherwise
           it is built from the x, y and z entries"""
        if transform is None:
            transform = np.zeros((3, 3))
            transform[0, 0] = x
            transform[1, 1] = y
            transform[2, 2] = z
        
        if verbose > 0:
            cell = self.cell.copy()

        """Transform the cell"""
        self.cell = np.matmul(self.cell, transform)

        if verbose > 0:
            for i in range(cell.shape[0]):
                string = "|%10.4f %10.4f %10.4f | %5s |%10.4f %10.4f %10.4f |"\
                    % (cell[i, 0], cell[i, 1], cell[i, 2], "  -->",\
                           self.cell[i, 0], self.cell[i, 1], self.cell[i, 2])

                if i == 0: print("=" * 35 + " " * 7 + "=" * 35)
                print(string)
                if i == 2: print("=" * 35 + " " * 7 + "=" * 35)

        """Transform positions back to cartesian coordinates"""
        self.dir2car()
        


    def getExtendedPositions(self, x = 0, y = 0, z = 0, idx = None,\
                           return_cart = True, verbose = 1):
        """Function for retreving an extended set of positions

        x = int, times to extend the cell

        y = int, times to extend the cell

        z = int, times to extend the cell

        idx = int, [int,], Index of atoms to include

        return_cart = bool, Return the positions in cartesian coordinates

        verbose = int, Print extra information
        """

        if idx is None: idx = np.arange(self.pos.shape[0])
        if isinstance(idx, (np.integer, int)): idx = np.arange([idx])

        """Total number of atoms"""
        N = self.pos.shape[0]

        """Change to direct coordinates"""
        self.car2dir()

        l = np.shape(idx)[0]

        xR = np.arange(0, x + 1, dtype = np.int)
        yR = np.arange(0, y + 1, dtype = np.int)
        zR = np.arange(0, z + 1, dtype = np.int)
        
        pos = self.pos.copy()[idx, :].T
        pos = np.tile(pos, (x + 1) * (y + 1) * (z + 1))

        n = 0
        for i in zR:
            for j in yR:
                for k in xR:
                    pos[:, l * n : l * (n + 1)] += np.array([k, j, i])[:, None]
                    n += 1
        
        cell = self.cell.copy()
        cell[:, 0] *= xR.shape[0]
        cell[:, 1] *= yR.shape[0]
        cell[:, 2] *= zR.shape[0]

        if return_cart:
            pos = np.matmul(self.cell, pos)
            return pos.T, cell
        else:
            return pos.T, cell



    def car2dir(self):
        """Change positions from cartesian to direct coordinates"""

        if self.pos_type.lower() == "d":
            return
        
        """c = M*d so here M(-1)*c = d"""
        self.pos = np.matmul(np.linalg.inv(self.cell), self.pos.T).T
        self.pos_type = "d"



    def dir2car(self):
        """Change positions from direct to cartesian coordinates"""

        if self.pos_type.lower() == "c":
            return
        
        """c = M*d"""
        self.pos = np.matmul(self.cell, self.pos.T).T
        self.pos_type = "c"


    def writeStructure(self, filename = None, format = "lammps", verbose = 1):
        """Function for writing the structure to specified file format

        filename = str(), Filename to write to

        format = str("lammps"/"vasp"/"eon"/"xyz"), Format to write to

        verbose = int, Print extra information
        """

        if filename is None: 
            if self.filename is None:
                filename = "Structure.%s" % format
            else:
                filename = self.filename

        """Write the structure object to specified file"""
        file_io.writeData(filename = filename, atoms = self, format = format, verbose = verbose - 1)
        
        if verbose > 0:
            string = "Structure written to file: %s (%s-format)" % (filename, format)
            ut.infoPrint(string)
