"""
Main document to run baseline optimization, experiments and tests.
"""
import os

from experimentation import baseline_optimization

if __name__ == '__main__':

    DEBUG = True
    if DEBUG:
        nfe = 5000
        epsilon_list = [1e-2, 1e-4, 1e-4, 1e-2, 1e-4, 1e-2, 1e-2]
        convergence_freq = 250
        description = "pwf"
        principle = "pwf"           # possible principles uwf, swf, pwf, gini
    else:
        # Access the environment variables for input parameters
        nfe = int(os.environ.get("NFE"))
        epsilon_list = [float(epsilon) for epsilon in os.environ.get("EPSILON_LIST").split()]
        convergence_freq = int(os.environ.get("CONVERGENCE_FREQ"))
        description = os.environ.get("DESCRIPTION")
        principle = os.environ.get("PRINCIPLE")

    # call the baseline optimization function 'run()' with the provided experiment input
    baseline_optimization.run(nfe, epsilon_list, convergence_freq, description, principle)
