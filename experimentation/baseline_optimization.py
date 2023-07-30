"""
Script for baseline optimization
"""
import random
import os
# import tarfile
# import shutil
from datetime import datetime

from ema_workbench import ema_logging, Model, RealParameter, ScalarOutcome, MultiprocessingEvaluator
from ema_workbench.em_framework.optimization import EpsilonProgress, ArchiveLogger
from experimentation.data_generation import generate_input_data
from model.model_nile import ModelNile



# def check_for_csv_files(log_directory):
#     csv_files = [file for file in os.listdir(log_directory) if file.endswith('.csv')]
#     return len(csv_files) > 0

# def extract_and_save_csv_from_tar_gz(log_directory):
#     tar_gz_file_path = f"{log_directory}/archives.tar.gz"

#     if not os.path.exists(tar_gz_file_path):
#         print(f"The file '{tar_gz_file_path}' does not exist.")
#         return

#     if check_for_csv_files(log_directory):
#         print("CSV files already exist in the directory.")
#     else:
#         with tarfile.open(tar_gz_file_path, "r:gz") as tar:
#             tar.extractall(path=log_directory)

#         csv_files = [file for file in os.listdir(log_directory) if file.endswith('.csv')]
#         if not csv_files:
#             print("No CSV files found in the archive.")
#         else:
#             print(f"CSV files saved: {len(csv_files)}")

def run(nfe:int, epsilon_list:list, convergence_freq:int, description:str, principle:str):
    """
    Perform baseline optimization using the EMA Workbench.

    Parameters:
    nfe (int): Number of function evaluations for the optimization.
    epsilon_list (list): List of epsilon values for the optimization.
    convergence_freq (int): Frequency of convergence logging during optimization.
    description (str): A string identifier for the experiment, used to label the output files.

    Returns:
    None

    Description:
    This function performs baseline optimization using the EMA Workbench framework.
    It uses the provided `nfe` (Number of Function Evaluations), `epsilon_list` (list
    of epsilon values), and `convergence_freq` (convergence logging frequency) to
    optimize the model. The results and convergence data are saved to CSV files.

    The `description` parameter is used as a string identifier to label the output files
    to make them easily identifiable and distinguishable for different experiments.

    The function sets up the model, levers, outcomes, convergence metrics, and other
    necessary configurations for the optimization process. It uses the `MultiprocessingEvaluator`
    for parallel evaluation and logs the optimization progress.

    The results of the optimization are saved in CSV files with filenames that include
    the experiment identifier to distinguish between different experiments. The filenames
    will have the format "baseline_results_description.csv" for the results and
    "baseline_convergence_description.csv" for the convergence data.

    Note:
    - Ensure that the ModelNile class, generate_input_data function, and the necessary
      EMA Workbench modules are properly imported before calling this function.
    - The output files will be saved in the "outputs/" directory relative to the script's location.
    """
    ema_logging.log_to_stderr(ema_logging.INFO)

    # creating or emptying the experiment folders
    output_directory = f"outputs/nfe{nfe}_{description}/"
    archive_directory = f"{output_directory}archive_logs"

    # if os.path.exists(output_directory):
    #     shutil.rmtree(output_directory)
    #     print(f"I emptied {output_directory}")

    # if os.path.exists(archive_directory):
    #     shutil.rmtree(archive_directory)
    #     print(f"I emptied {archive_directory}")

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(archive_directory, exist_ok=True)

    nile_model = ModelNile(principle=principle)
    nile_model = generate_input_data(nile_model, sim_horizon=20)

    em_model = Model("NileProblem", function=nile_model)

    parameter_count = nile_model.overarching_policy.get_total_parameter_count()
    n_inputs = nile_model.overarching_policy.functions["release"].n_inputs
    n_outputs = nile_model.overarching_policy.functions["release"].n_outputs
    # RBF_count = nile_model.overarching_policy.functions["release"].RBF_count
    p_per_RBF = 2 * n_inputs + n_outputs

    lever_list = list()
    for i in range(parameter_count):
        modulus = (i - n_outputs) % p_per_RBF
        if (
            (i >= n_outputs)
            and (modulus < (p_per_RBF - n_outputs))
            and (modulus % 2 == 0)
        ):  # centers:
            lever_list.append(RealParameter(f"v{i}", -1, 1))
        else:  # linear parameters for each release, radii and weights of RBFs:
            lever_list.append(RealParameter(f"v{i}", 0, 1))

    em_model.levers = lever_list

    # specify outcomes
    em_model.outcomes = [
        ScalarOutcome("egypt_irr", ScalarOutcome.MINIMIZE),
        ScalarOutcome("egypt_90", ScalarOutcome.MINIMIZE),
        ScalarOutcome("egypt_low_had", ScalarOutcome.MINIMIZE),
        ScalarOutcome("sudan_irr", ScalarOutcome.MINIMIZE),
        ScalarOutcome("sudan_90", ScalarOutcome.MINIMIZE),
        ScalarOutcome("ethiopia_hydro", ScalarOutcome.MAXIMIZE),
        ScalarOutcome("principle_result", ScalarOutcome.MAXIMIZE),
    ]

    convergence_metrics = [
        EpsilonProgress(),
        ArchiveLogger(
            archive_directory,
            [lever.name for lever in em_model.levers],
            [outcome.name for outcome in em_model.outcomes],
        ),
    ]

    random.seed(123)
    before = datetime.now()

    with MultiprocessingEvaluator(em_model) as evaluator:
        results, convergence = evaluator.optimize(
            nfe=nfe,
            searchover="levers",
            epsilons=epsilon_list,
            convergence_freq=convergence_freq,
            # real convergence_freq=500,
            convergence=convergence_metrics,
        )

    after = datetime.now()


    with open(f"{output_directory}time_counter_{description}.txt", "w") as f:
        f.write(
            f'''experiment {description} took {after-before} time to do {nfe} NFEs with 
            a convergence frequency of {convergence_freq} and epsilons: {epsilon_list}, for principle {principle}.'''
        )

    # # unpack logs
    # extract_and_save_csv_from_tar_gz(archive_directory)

    # Use description in the filename for the CSV files
    results_filename = f"{output_directory}baseline_results_nfe{nfe}_{description}.csv"
    convergence_filename = f"{output_directory}baseline_convergence_nfe{nfe}_{description}.csv"

    results.to_csv(results_filename)
    convergence.to_csv(convergence_filename)
