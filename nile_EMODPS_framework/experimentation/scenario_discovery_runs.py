# Script for open exploration/scenario discovery


import numpy as np
import os
import pandas as pd
import sys

from datetime import datetime


from ema_workbench import RealParameter, ScalarOutcome, Model, Policy
from ema_workbench import MultiprocessingEvaluator, ema_logging

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from model.model_nile import ModelNile


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    output_directory = "../outputs/"
    nile_model = ModelNile()

    lever_count = nile_model.overarching_policy.get_total_parameter_count()

    em_model = Model("NileProblem", function=nile_model)
    em_model.uncertainties = [
        RealParameter("yearly_demand_growth_rate", 0.1, 0.3),
        RealParameter("blue_nile_mean_coef", 0.75, 1.25),
        RealParameter("white_nile_mean_coef", 0.75, 1.25),
        RealParameter("atbara_mean_coef", 0.75, 1.25),
        RealParameter("blue_nile_dev_coef", 0.5, 1.5),
        RealParameter("white_nile_dev_coef", 0.5, 1.5),
        RealParameter("atbara_dev_coef", 0.5, 1.5)
    ]
    em_model.levers = [
        RealParameter("v" + str(i), 0, 1) for i in range(lever_count)
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

    n_scenarios = 4000
    policy_df = pd.read_csv(f"{output_directory}policies_exploration.csv")
    my_policies = [
        Policy(
            policy_df.loc[i, "name"],
            **(policy_df.iloc[i, :-1].to_dict())
        ) for i in policy_df.index
    ]

    before = datetime.now()

    with MultiprocessingEvaluator(em_model) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(
            n_scenarios,
            my_policies
        )

    after = datetime.now()

    with open(f"{output_directory}time_counter_open_exp.txt", "w") as f:
        f.write(
            f"It took {after-before} time to run {n_scenarios} scenarios {len(my_policies)} policies"
        )
    outcomes = pd.DataFrame.from_dict(outcomes)
    experiments.to_csv(f"{output_directory}experiments_exploration.csv")
    outcomes.to_csv(f"{output_directory}outcomes_exploration.csv")
