This is code corresponding to the manuscript "Single-Cell and Population-Level Neuromodulation Dynamics in Dual-Electrode Intracortical Stimulation", https://doi.org/10.1101/2025.06.25.661479

The simulation is run using `python3 plot_vc_output.py`<br>
If you would like to run multiple simulations in a row using different seeds, use the bash script `run_trials_provide_seed.sh`

In order to run this code, you need to install NEST and NESTML. This simulation software can be installed on your OS using a conda environment. Follow these steps:<br>
Install miniconda - https://docs.anaconda.com/miniconda/<br>
Install NEST within the conda environment - https://nest-simulator.readthedocs.io/en/stable/installation/conda_forge.html#conda-forge-install/<br>
Install NESTML within the same conda environment - https://nestml.readthedocs.io/en/latest/installation.html<br>
Install the Adaptive Exponential Integrate and Fire neuron with a beta function synaptic response (aeif_cond_beta) within your NESTML installation (instructions below).

From the "installer_files" git folder:<br>
Copy the "aeif_cond_beta.nestml" file into the path that was created to the NESTML models within your environment. The path will look similar to: .../.../miniconda3/envs/MY_ENVIRONMENT_NAME/models/neurons<br>
Copy the "create_aeif_cond_beta.py" to a relevant folder on your computer.<br>
Create the aeif_cond_beta neuron by running the python script. Before running the script, update the "input_path" and "target_path" in the Python file. The input path should be the path where you copied the "aeif_cond_beta.nestml" file.
