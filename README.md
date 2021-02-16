# interfaceBuilder

Package for building and analyzing atomic interfaces. The process of finding interfaces
is based on the approach described in J. Phys.: Condens. Matter 29 (2017) 185901 (7pp)[1]
with added freedom to construct any specific match of cell paramters or restrain the search
to specific paramters of interest. Built to allow large scale sreening of interfaces
by making it easy to work with entire sets of discovered interface matches.
Beyond this the package contains a large number of tools
to analyze and evaluate the interface data set as well as allow easy exporting of interfaces
and the corresponding surfaces to common simulation file formats (VASP,LAMMPS,EON). For furter
analysis paramteres calculated elsewere such as work of separation and interfacial energy can be 
loaded back into the data set and combined with the original interface data. 

This project is licensed under the terms of the MIT license.

## Contains the following files

----------------------------
file_io   - Reading and writing of files  
structure - Object for holding cell geometries and information  
interface - Object for holding interface collections  
utils     - Utility functions used by the other classes  
inputs    - User defined manual database for holding structure information  
style     - Some plot defaults  

Structures - Folder with some example structures that can be loaded  
Scripts    - Folder with a few scripts for some of the basic operations  

----------------------------

## Used packages

----------------------------
The following python packages are used in this package with the version used in the creation
indicated.

numpy (1.18.1)  
scipy (1.3.2)  
pandas (1.0.0)  
matplotlib (3.3.2)  

## Example operations

They easiest way of working with the package and setting up interface combinations
and analyzing data is from an ipython terminal. To get all base packages needed 
to run this package use e.g. the anaconda distribution where all are included.
Add the package to you path or start the ipython terminal in the folder where 
interfaceBuilder is keept.
 
To test building a collection of interfaces do the following
Import the module and the pieces needed
```
import interfaceBuilder as ib
```
Then build two structures by writing
```
a = ib.structure.Structure(load_from_input = "W100_L")
b = ib.structure.Structure(load_from_input = "W110_L")
```
This creates two structure files (a,b) containing the geometric information about
the a cell (100-W) and the b cell (110-W). Load_from_input is just a shortcut 
to load the predefined cell information contained in the inputs file. Complete
reading of input-files from usefull formats e.g. VASP/LAMMPS/EON is done be specifying
structure.Structure(load_from_file = filename, format = format)
```
i = ib.interface.Interface(structure_a = a, structure_b = b)
```
This creates the inital interface structure, but it is still empty at this point.
```
i.matchCells()
```
This creates all interface cell matches as specified by the various default input options. To check the defaults
and the use of any function type the method with a ? to display the docstring.
```
i.matchCells?
i.printInterfaces(idx = range(50))
```
Prints the first 50 interfaces as currently sorted
```
i.sortInterfaces(sort = "eps_mas")
i.printInterfaces(idx = range(50))
```
Sorts the interfaces based on the mean absolute strain as defined in [1]. 
And again prints the first 50 interfaces.
```
i.plotInterface(idx = 0)
```
Plots the basis vectors of the first interface (as currently sorted) with base latticies as background.
```
i.plotCombinations()
```
Plots mean absolute strain against nr of atoms in the interface or all interfaces
```
C, E = i.getAtomStrainMatches(matches = 2500)
i50 = i.getAtomStrainIdx(matches = 50)
i.plotCombinations(const = C, exp = E, mark = i50)
```
Plots the same combination plot as before but with all interfaces above/below
<nr_atoms> - C * |strain| ** E ratio separated 
and the interfaces with indices in array i50 highlighted.
```
i.removeByAtomStrain(keep = 5000)
```
Removes all interaces except for 5000 interfaces below the same ratio as above but with the parameters adjusted
to include 5000 interfaces.
```
i.plotInterface(idx = 5, align_base = "no")
```
Plots the cell vectors of interface with index 5, both top surface and bottom surfaces displayed just as 
it was found when matched. 
```
i.summarize(idx = 5)
```
Plots a summery in 4 plots of the specified interface.
```
i.exportInterface(idx = 5, d = 2.3, z_1 = 2, z_2 = 4, vacuum = 10, verbose = 2, format = "lammps")
```
Exports the interface with current index 5 to a lammps data file with a 10 Å vacuum above the top surface. A distance of 2.3 Å between the bottom and top cell and with 2 repetitions of the bottom cell (cell 1) and 4 repetitions of the top cell (cell 2) in the z directions.
```
i.exportSurface(idx = 10, z = 3, verbose = 2, format = "vasp", surface = 2, strained = False, vacuum = 12)
```
Exports the top (cell 2) surface from the interface with index 10 in a vasp poscar format. The surface is repeated 3 times in the z direction and has a vacuum region above of 12 Å, it is exported in the unstrained state i.e. not as it would be if exported along with the full interface. That can be switched by the strain True/False parameter.
```
i.plotProperty(x = "angle", y  = "eps_mas", idx = range(500))
```
Plots the angle against the mean absolut strain or the first 500 interfaces as currently sorted.
```
i.plotProperty(x = "angle", y  = "eps_mas", z = "density", idx = range(500), colormap = "plasma")
```
Again plots the angle against the mean absolute strain but with the nr of atoms / area in the top interface surface (the strained one) in comparison 
to the atoms / area in the unstrained case, above 1 denser than usual below 1 less dense then usual. Using the plasma colormap.

Write i.plotProperty? to see supported plot options. And write i.getData? to see all available keywords

To add an alternative base to reflect a relaxed bottom surface when switching form e.g. LAMMPS --> VASP do the following.
It will load the relaxed structures from the POSCAR type files stored in the structures folder. The new base is always added in a pair 
of the [bottom, top] surfaces respectively and adding a new one overwrites the old. To export a structure using the alternative base
add the keyword ab = True to the exportInterface or exportStructure functions.
```
i.addAltBase(from_file = [<path_to_bottom_structure>, <path_to_top_structure>], format = "vasp")
```
To display the difference in strain due to the different basis do the following to show the 50 first interfaces as currently sorted
and add the difference between the two as an extra y axis. ab = [1] specifies that the variable at index 1 should use the alternative
base. This workes for any property that is affected by the change in base, e.g. area, cell vectors etc. Available variables for var are
the same as for the plot properties function. 
```
i.compareInterfaces(var = ["eps_mas", "eps_mas"], delta = True, ab = [1], idx = range(50))
```
The plot property function can specify several x/y and x/y/z pairs as well and can also use the ab keyword to plot the specified index
using the alternative base if such a base have been added.
```
i.plotProperty(x = ["angle", "angle"], y = ["eps_mas", "eps_mas"], z = ["density", "density"], ab = [1], idx = range(500), colormap = "plasma", m = ["o", "^"], ms = 5)
```
```
someValues = np.random.random(500)
i.plotProperty(x = "angle", y  = "other", other = someValues, z = "density", idx = range(500), colormap = "plasma")
```
Plots angle against the custom set of values contained in the someValues array and supplied with the other keyword with the density once again colorcoded to the colormap. This is to allow any specially calculated property to be easily ploted against other paramerters. The length of the custom data must match the length of the idx parameter or the total length of the interface dataset if idx is omitted.
```
i.matchCells(M = [2, 0], N = [0, 3], target = [[2, 0], [1, 3]])
```
Build the exact interface that takes M times the first cell vector in the top cell and N times the second
cell vector in the top cell and mathes it against the target combination of the bottom cell vectors. Take care when using this functionality as only right handed combinations are keept, switch the N and M permutations 
or the target permutaions if the program yells about it.
```
i.plotInterface(idx = 0, align_base = "no")
```
Displays the single created interface.