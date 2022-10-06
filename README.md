# tvb-wongwang-ms
Repository containing the code used to process the data and obtain the results in "Using The Virtual Brain to study the relationship between structural and functional connectivity in patients with multiple sclerosis: a multicentre study".

This repository implements scripts to run Wong Wang based computational models, performing, for a given subject with associated Functional (FC) and Structural (SC) Connectivity, a parameter sweep to obtain the best fit between the FC and the simulated FC. 

## Warning
Code is provided "as is", an should not be used out of the box, and probably needs a bit of tinkering to work with your own data. However, I believe it is more useful to share the code this way than not sharing it at all.

Code has been designed to work in an HPC environment. Can be adapted to work locally.

## Implementation
Code has worked with Python 3.7+ 
gcc 8.4.0 compiler for the C code
Code uses optuna (https://optuna.readthedocs.io/en/stable/) for saving the results and optimization of each subject and easily store it. Optimization is done
manually.

## Table of contents
* analysis/: Various scripts to analyze the results and the optimization.
* fast_tvb/: C code of the model. Refer to README.MD inside there and Credits here for authoring credits. Various extra models are included.
* hpc/: Example script to run the optimization in the HPC code. 
* src/: Various files and scripts that are called by the main programs
* hpc_get_results.py: Run over an existing subject, without actually running the simulations again.
* hpc_run_all.py: For a csv file with N subjects, creates an HPC task for each subjects and runs them in parallel.
* hpc_run_subj.py: Run a single subject. Compiles the C code with the selected parameters and performs the parameter search.


## Credits
For any issues or questions about the code, contact Gerard Martí-Juan (gerardmartijuan(at)gmail(dot)com)

Regarding the C implementation of the model, refer to:
Schirner, M., McIntosh, A. R., Jirsa, V., Deco, G., & Ritter, P. (2018). Inferring multi-scale neural mechanisms with brain network modelling. Elife, 7, e28927.
By Michael Schirner (michael.schirner@charite.de) or Petra Ritter (petra.ritter@charite.de).
Original repository: https://github.com/BrainModes/fast_tvb