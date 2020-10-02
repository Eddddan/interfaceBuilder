# interfaceBuilder

Scripts used to build interfaces from initial atomistic structures.
The approach is the one developed by the authors in 
[1] J. Phys.: Condens. Matter 29 (2017) 185901 (7pp)

Beyond interface discovery and generation a lot of functionality
for analyzing the discovered interfaces are included.
 
To test building a collection of interfaces do the following

Import the module and the pieces needed
 
from interfaceBuilder import structure
from interfaceBuilder import interface

Then build two structures by writing

a = structure.Structure(load_from_input = "WC0001_L")

b = structure.Structure(load_from_input = "W100_L")

This creates two structure files (a,b) containing the geometric information about
the a cell (0001-WC) and the b cell (100-W). Load_from_input is just a shortcut 
to load the predefined cell information contained in the inputs file. Complete
reading of input-files from usefull formats e.g. VASP/LAMMPS/EON is done be specifying
structure.Structure(load_from_file = filename, format = format)

i = interface.Interface(structure_a = a, structure_b = b)

This creates the inital interface structure, but it is still empty at this point.

i.matchCells()

This creates all interface cell matches as specified by the various default input options.

i.printInterfaces(idx = range(50))

Prints the first 50 interfaces as currently sorted

i.sortInterfaces(sort = "eps_mas")

i.printInterfaces(idx = range(50))

Sorts the interfaces based on the mean absolute strain as defined in [1]. 
And again prints the first 50 interfaces.

i.plotInterface(idx = 0)

Plots the basis vectors of the first interface (as currently sorted) with base latticies as background.

i.plotCombinations()

Plots mean absolute strain against nr of atoms in the interface or all interfaces

i.summerize(idx = 5)

Plots a summery in 4 plots of the specified interface.

i.exportInterface(idx = 5, d = 2.3, z_1 = 2, z_2 = 4, vacuum = 10, verbose = 2, format = "lammps")

Exports the interface with current index 5 to a lammps data file with a 10 Å vacuum above the top surface. A distance of 2.3 Å between the bottom and top cell and with 2 repetitions of the bottom cell (cell 1) and 4 repetitions of the top cell (cell 2) in the z directions.

i.exportSurface(idx = 10, z = 3, verbose = 2, format = "vasp", surface = 2, strained = False, vacuum = 12)

Exports the top (cell 2) surface from the interface with index 10 in a vasp poscar format. The surface is repeated 3 times in the z direction and has a vacuum region above of 12 Å, it is exported in the unstrained state i.e. not as it would be if exported along with the full interface. That can be switched by the strain True/False parameter.

i.plotProperty(x = "angle", y  = "eps_mas", idx = range(500))

Plots the angle against the mean absolut strain or the first 500 interfaces as currently sorted.

i.plotProperty(x = "angle", y  = "eps_mas", z = "density", idx = range(500), colormap = "plasma")

Again plots the angle against the mean absolute strain but with the nr of atoms / area in the top interface surface (the strained one) in comparison 
to the atoms / area in the unstrained case, above 1 denser than usual below 1 less dense then usual. Using the plasma colormap.

Write i.plotProperties? to see supported plot options.

someValues = np.random.random(500)

i.plotProperty(x = "angle", y  = "other", other = someValues, z = "density", idx = range(500), colormap = "plasma")

Plots angle against the custom set of values contained in the someValues array and supplied with the other keyword with the density once again colorcoded to the colormap. This is to allow any specially calculated property to be easily ploted against other paramerters. The length of the custom data must match the length of the idx parater or the total length of the interface dataset if idx is omitted.
