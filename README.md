# general
A collection of python Scripts used for the PMT simulation as part of my Bachelor thesis.
See the [Bachelor thesis](https://www.institut3b.physik.rwth-aachen.de/global/show_document.asp?id=aaaaaaaaaawzven) for a more detailed description of whats going on.
# requirements
This project has been written for python 3.6.
It will most likely work with older versions of python 3.
The core module (pmt_sim) works on old versions of python 2, too.

Dependencies: numpy, matplotlib
# files
### pmt_sim.py
The main module that contains the actual simulation functions.
### int.py
GUI application that allows to quickly test different simulation settings.
Allows to load measured data
### example.py
Minimalistic script to run the simulation
### helpers.py
Module that contains a function to load the data from measurement files.
### simfit.py
Finds a local best fit for the simulation parameters to a measurement.
### oldcounter.py
Partial reimplementation of the collections.Counter class.
It is used by the pmt_sim module because of the old python version on the CIP-Pool.
