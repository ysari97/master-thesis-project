"""
Main document to run baseline optimization, experiments and tests.
"""
import os

from experimentation import baseline_optimization

if __name__ == '__main__':
    # Access the environment variables for input parameters
    nfe = int(os.environ.get("NFE", 2))
    epsilon_list = [float(epsilon) for epsilon in os.environ.get("EPSILON_LIST", "0.01 0.001 0.001 0.01 0.001 0.01").split()]
    convergence_freq = int(os.environ.get("CONVERGENCE_FREQ", 1))
    experiment = os.environ.get("EXPERIMENT", "my_experiment")

    # call the baseline optimization function 'run()' with the provided experiment input
    baseline_optimization.run(nfe, epsilon_list, convergence_freq, experiment)
