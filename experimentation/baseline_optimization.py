"""
Script for baseline optimization
"""
import random
from datetime import datetime

from ema_workbench import ema_logging, Model, RealParameter, ScalarOutcome, MultiprocessingEvaluator
from ema_workbench.em_framework.optimization import EpsilonProgress, ArchiveLogger
from experimentation.data_generation import generate_input_data
from model.model_nile import ModelNile


def run():
    ema_logging.log_to_stderr(ema_logging.INFO)

    output_directory = "outputs/"
    nile_model = ModelNile()
    nile_model = generate_input_data(nile_model, sim_horizon=20)

    em_model = Model("NileProblem", function=nile_model)

    parameter_count = nile_model.overarching_policy.get_total_parameter_count()
    n_inputs = nile_model.overarching_policy.functions["release"].n_inputs
    n_outputs = nile_model.overarching_policy.functions["release"].n_outputs
    RBF_count = nile_model.overarching_policy.functions["release"].RBF_count
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
    ]

    convergence_metrics = [
        EpsilonProgress(),
        ArchiveLogger(
            f"{output_directory}archive_logs",
            [lever.name for lever in em_model.levers],
            [outcome.name for outcome in em_model.outcomes],
        ),
    ]

    # real nfe = 50000
    nfe = 2
    epsilon_list = [1e-2, 1e-3, 1e-3, 1e-2, 1e-3, 1e-2]

    random.seed(123)
    before = datetime.now()

    with MultiprocessingEvaluator(em_model) as evaluator:
        results, convergence = evaluator.optimize(
            nfe=nfe,
            searchover="levers",
            epsilons=epsilon_list,
            convergence_freq=1,
            # real convergence_freq=500,
            convergence=convergence_metrics,
        )

    after = datetime.now()

    with open(f"{output_directory}time_counter.txt", "w") as f:
        f.write(
            f"It took {after-before} time to do {nfe} NFEs with epsilons: {epsilon_list}"
        )

    results.to_csv(f"{output_directory}baseline_results.csv")
    convergence.to_csv(f"{output_directory}baseline_convergence.csv")
