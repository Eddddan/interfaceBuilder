
import inputs
import file_io
import utils as ut
import numpy as np
import matplotlib.pyplot as plt


"""Set some plotting defaults"""
plotParams = {"font.size": 10, "font.family": "serif", "axes.titlesize": "medium",\
              "axes.labelsize": "small", "axes.labelweight": "normal",\
              "axes.titleweight": "semibold", "legend.fontsize": "small",\
              "xtick.labelsize": "small", "ytick.labelsize": "small",\
              "figure.titlesize": "medium", "figure.titleweight": "semibold",\
              "lines.linewidth": 1, "lines.marker": "None", "lines.markersize": 2,\
              "lines.markerfacecolor": "None"}
plt.rcParams.update(**plotParams)


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
                type_n[:] = "H"
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


    def writeStructure(self, filename = None, format = "lammps"):
        """Function for writing the structure to a file"""

        if filename is not None:
            file_io.writeData(filename = filename, atoms = self, format = format)
        else:
            file_io.writeData(filename = self.filename, atoms = self, format = format)


    def printStructure(self):

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
                                                        self.type_i[i], self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]))

        print("-" * len(string))
        string = "Nr of Atoms: %i | Nr of Elements: %i" % (self.pos.shape[0], np.unique(self.type_i).shape[0])
        print(string)



    def alignStructure(self, dim = [1, 0, 0], align = [1, 0, 0], verbose = 1):
        """Function for aligning a component of the structure in a specific dimension

        dim = [float, float, float], the cell will be aligned to have this
        cell vector entierly in the direction of the cartesian axis supplied
        in the align parameter. Dim in direct coordinates.
                          
        align = [float, float, float], cartesian axis to align dim to. Align in cartesian coordinates.
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


    def getNeighborDistance(self, idx = None, r = 6, idx_to = None,\
                            extend = np.array([1, 1, 1], dtype = bool),\
                            verbose = 1):
        """Function for getting the distance between specified atoms within
           radius r"""

        """Check some defaults"""
        if idx is None: idx = np.arange(self.pos.shape[0])
        if isinstance(idx, (int, np.integer)): idx = np.array([idx])
        if idx_to is None: idx_to = np.arange(self.pos.shape[0])
        if isinstance(idx_to, (int, np.integer)): idx_to = np.array([idx_to])
        extend = np.array(extend, dtype = bool)

        """Cell extension to comply with the sepecified r value"""
        box = self.getBoxLengths()
        rep = np.ceil(r / box) - 1
        rep = rep.astype(np.int)
        rep[np.logical_not(extend)] = 0

        """If the box < r then extend the box. Otherwise wrap the cell"""
        if np.any(box < r):

            if verbose > 0:
                string = "Replicating cell by %i,%i,%i (x,y,z)"\
                         % (rep[0] + 1, rep[1] + 1, rep[2] + 1)
                ut.infoPrint(string)

            pos_to, cell = self.getExtendedPositions(x = rep[0], y = rep[1], z = rep[2],\
                                                  idx = idx_to, return_cart = True,\
                                                  verbose = verbose - 1)

            """Change to cartesian coordinates"""
            self.dir2car()

            """Change back to direct coordinates using the new extended cell"""
            pos_to = np.matmul(np.linalg.inv(cell), pos_to.T).T
            pos_from = np.matmul(np.linalg.inv(cell), self.pos) 
        else:
            if verbose > 0:
                string = "Cell is only wrapped, not extended"
                ut.infoPrint(string)

            """Change to direct coordinates"""
            self.car2dir()
            pos_to = self.pos.copy()[idx_to, :]
            pos_from = self.pos.copy()
            cell = self.cell

        ps = pos_to.shape[0]
        dist = np.zeros((np.shape(idx)[0] * ps, 3))

        for i, item in enumerate(idx):
            d = pos_to - pos_from[item, :]
        
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



    def getRDF(self, idx = None, idx_to = None, r = 6, dr = 0.1, bins = None,\
               extend = np.array([1, 1, 1], dtype = bool), edges = False,\
               verbose = 1):
        """Function for getting the radial distribution function around and 
        to specified atoms"""

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



    def plotRDF(idx = None, idx_to = None, r = 6, dr = 0.1, bins = None, edges = False,\
                extend = np.array([1, 1, 1], dtype = bool), verbose = 1):
        """Function for ploting the RDF and cumulative distribution"""

        print("Plot RDF")

    


    def extendStructure(self, x = 1, y = 1, z = 1, reset_index = False, verbose = 1):
        """Function for repeating the cell in x, y or z direction"""

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



    def getExtendedPositions(self, x = 0, y = 0, z = 0, idx = None,\
                           return_cart = True, verbose = 1):
        """Function for retreving an extended set of positions"""

        if idx is None: idx = np.arange(self.pos.shape[0])
        if isinstance(idx, (np.integer, int)): idx = np.arange([idx])

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
        """Function for writing the structure to specified file format"""

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
