# interfaceBuilder

Scripts used to build interfaces from initial atomistic structures.
The approach is the one developed by the authors in 
[1] J. Phys.: Condens. Matter 29 (2017) 185901 (7pp)
 
To test building a collection of interfaces do the following

a = structure.Structure(load_from_input = "WC0001_L")

b = structure.Structure(load_from_input = "W100_L")

This creates two structure files (a,b) containing the geometric information about
the a cell (0001-WC) and the b cell (100-W). Load_from_input is just a shortcut 
to load the predefined cell information contained in the inputs file. Complete
reading of input-files from usefull formats e.g. VASP/LAMMPS/EON etc. is beeing
added as needed.

i = interface.Interface(structure_a = a, structure_b = b)

This creates the inital interface structure, but it is still empty at this point.

i = matchCells()

This creates all interface cell matches as specified by the various input options.

i.printInterface(idx = range(50))

Prints the first 50 interaces

i.sortInterface(sort = eps_mas)

i.printInterfaces(idx = range(50))

Sorts the interfaces based on the mean absolute strain as defined in [1]. 
And again prints the first 50 interfaces.

i.plotInterface(idx = 0)

Plots the basis vectors of the first interface (as currently sorted) with base latticies as background.

i.plotCombinations()

Plots mean absolute strain against nr of atoms in the interface or all interfaces
