
import inputs
import file_io
import numpy as np


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
            cell, pos, type_n, idx = file_io.readData(load_from_file, format)

            if np.all(pos >= 0) and np.all(pos <= 1):
                pos_type = "d"
            else:
                pos_type = "c"

            type_i = np.zeros(type_n.shape[0])
            for i, item in enumerate(np.unique(type_n)):
                type_i[type_n == item] = i + 1

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
        elements = [ "H", "He", "Li", "Be",  "B",  "C",  "N",  "O",  "F",\
                    "Ne", "Na", "Mg", "Al", "Si",  "P",  "S", "Cl", "Ar"]
        if type_n is None:
            if type_i is None:
                type_i = np.ones(pos.shape[0])
                type_n = np.chararray(pos.shape[1], itemsize = 2)
                type_n[:] = "H"
            else:
                type_n = np.chararray(pos.shape[1], itemsize = 2)
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


    def writeStructure(self, filename = None, format = None):
        """Function for writing the structure to a file"""

        if format is None: format = "lammps"

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
        file_io.writeData(filename = filename, atoms = self, format = format)
        
        if verbose > 0:
            string = "Structure written to file: %s (%s-format)" % (filename, format)
            print("=" * len(string))
            print(string)
            print("=" * len(string))
