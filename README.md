# iFlow
The iFlow modelling framework allows for a systematic analysis of the water motion and sediment transport processes in estuaries and tidal rivers and the sensitivity of these processes to model parameters. iFlow has a modular structure, making the model easily extendible. This allows one to use iFlow to construct anything from very simple to rather complex models. 

The iFlow core is designed to make it easy to include, exclude or change model components, called modules. The core automatically ensures modules are called in the correct order, inserting iteration loops over groups of modules that are mutually dependent. The iFlow core also ensures a smooth coupling of modules using analytical and numerical solution methods or modules that use different computational grids.

iFlow includes a range of modules for computing the hydrodynamics and suspended sediment dynamics in estuaries and tidal rivers. These modules employ perturbation methods, which allow for distinguishing the effect of individual forcing terms in the equations of motion and transport. Also included are several modules for computing turbulence and salinity. These modules are supported by auxiliary modules, including a module that facilitates sensitivity studies. 

Additional to an explanation of the model functionality, we present two case studies, demonstrating how iFlow facilitates the analysis of model results, the understanding of the underlying physics and the testing of parameter sensitivity. A comparison of the model results to measurements show a good qualitative agreement.   
 *From paper abstract Dijkstra et al (2017)*

# Read more and getting started
Several options are provided to learn to work with iFlow: 
* A getting started tutorial - see folder 'Examples/Tutorial'
* Example input files linked to (scientific) publications or other applications of the model - see folder 'Examples'
* Extensive manuals for iFlow containing an in-depth description of the core of iFlow and the modules included - see folder 'Manuals'. 
NB. The manuals and tutorial documents have not yet been adapted for iFlow v3.*

Our paper "The iFlow Modelling Framework v2.4. A modular idealized process-based model for flow and transport in estuaries" is published and freely available via [this link](https://doi.org/10.5194/gmd-10-2691-2017).

# Programming language, installation and prerequisites
iFlow v3.* works in Python 3.7, 3.8 and 3.9 (older versions iFlow v2.* work in Python 2.7). To install, make sure python and common required packages are installed. We recommend installing Anaconda, which includes Python and all packages needed to run iFlow. For more on installation, read the manual or the tutorial.

Running iFlow requires no or very little knowledge of Python or programming in general. Users will however quickly find themselves wanting to make custom visualisations (figures/tables), requiring basic knowledge of Python. 

# License and terms of use
When using iFlow in any scientific publication, technical report or otherwise formal writing, please cite our paper:  
Dijkstra, Y.M., Brouwer, R.L, Schuttelaars, H.M. and Schramkowski, G.P., 2017, "The iFlow Modelling Framework v2.4. A modular idealized process-based model for flow and transport in estuaries." Geoscientific Model Development, 10, 2691-2713.

The iFlow code is property of the Flemish Dutch Scheldt Committee (VNSC) and is licensed under LGPL (GNU Lesser General Public License). In summary, this means that the code is open source and may be used freely for non-commercial and commercial purposes. Any alterations to the iFlow source code (core and modules) must be licensed under LGPL as well. However, new modules or a coupling between iFlow and other software may be published under a different licence. See the [LICENSE.md](LICENSE.md) for all details. Users of iFlow are encouraged to make their own developed modules and model applications open source as well. 

# Authors
* Yoeri Dijkstra - Main developer, responsible for the iFlow core and numerical, analytical and general module packages [(GitHub profile)](https://github.com/YoeriDijkstra) 
* Ronald Brouwer - Main developer, responsible for the semi-analytical module package [(GitHub profile)](https://github.com/rlbrouwer) 

# Contributing
Contributions to iFlow are welcome. Contact Yoeri Dijkstra through GitHub for more information.

# Acknowledgments
The initial developement of iFlow was funded by VNSC (http://www.vnsc.eu) through contracts 3109 6925 and 3110 6170 of the "Agenda for the Future" scientific research program that is aimed at a better understanding of the Scheldt Estuary for improved policy and management. 

