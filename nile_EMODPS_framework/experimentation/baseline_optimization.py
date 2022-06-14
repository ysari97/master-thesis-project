# Script for baseline optimization


import numpy as np
import os
import pandas as pd
import sys

from datetime import datetime


from ema_workbench import RealParameter, ScalarOutcome, Constant, Model
from ema_workbench import MultiprocessingEvaluator, ema_logging
from ema_workbench.em_framework.optimization import (
    EpsilonProgress,
    ArchiveLogger,
)

from data_generation import generate_input_data
from wrapper import model_wrapper
from nile_EMODPS_framework.model.model_nile import ModelNile

# from model_nile import ModelNile

if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    output_directory = "../Outputs/"
    nile_model = ModelNile()
    model_object = generate_input_data(nile_model, sim_horizon=20)

    parameter_count = nile_model.overarching_policy.get_total_parameter_count()

    em_model = Model("NileProblem", function=nile_model)
    em_model.levers = [
        RealParameter("v" + str(i), 0, 1) for i in range(parameter_count)
    ]

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

    nfe = 50000
    epsilon_list = [1e2, 1e1, 1e-2, 1e2, 1e1, 1e3]

    before = datetime.now()

    with MultiprocessingEvaluator(em_model) as evaluator:
        results, convergence = evaluator.optimize(
            nfe=nfe,
            searchover="levers",
            epsilons=epsilon_list,
            convergence_freq=500,
            convergence=convergence_metrics,
        )

    after = datetime.now()

    with open(f"{output_directory}time_counter.txt", "w") as f:
        f.write(
            f"It took {after-before} time to do {nfe} NFEs with epsilons: {epsilon_list}"
        )

    results.to_csv(f"{output_directory}baseline_results.csv")
    convergence.to_csv(f"{output_directory}baseline_convergence.csv")