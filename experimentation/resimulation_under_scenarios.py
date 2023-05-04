# Script for open exploration/scenario discovery


import numpy as np
import os
import pandas as pd
import sys

from datetime import datetime


from ema_workbench import RealParameter, ScalarOutcome, Model, Policy, Scenario
from ema_workbench import MultiprocessingEvaluator, ema_logging

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
from model.model_nile_scenario import ModelNileScenario


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)

    output_directory = "../outputs/"
    nile_model = ModelNileScenario()

    em_model = Model("NileProblem", function=nile_model)
    em_model.uncertainties = [
        RealParameter("yearly_demand_growth_rate", 0.01, 0.03),
        RealParameter("blue_nile_mean_coef", 0.75, 1.25),
        RealParameter("white_nile_mean_coef", 0.75, 1.25),
        RealParameter("atbara_mean_coef", 0.75, 1.25),
        RealParameter("blue_nile_dev_coef", 0.5, 1.5),
        RealParameter("white_nile_dev_coef", 0.5, 1.5),
        RealParameter("atbara_dev_coef", 0.5, 1.5),
    ]

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

    fixed_uncertainties = {
        "atbara_mean_coef": 1,
        "blue_nile_dev_coef": 1,
        "white_nile_dev_coef": 1,
        "atbara_dev_coef": 1,
    }
    my_scenarios = [
        Scenario(
            "Baseline",
            yearly_demand_growth_rate=0.02,
            blue_nile_mean_coef=1,
            white_nile_mean_coef=1,
            uniform_flag=1,
            **fixed_uncertainties,
        ),
        Scenario(
            "OptimScen",
            yearly_demand_growth_rate=0.02,
            blue_nile_mean_coef=1,
            white_nile_mean_coef=1,
            uniform_flag=-1,
            **fixed_uncertainties,
        ),
        Scenario(
            "HighD_LowB",
            yearly_demand_growth_rate=0.03,
            blue_nile_mean_coef=0.75,
            white_nile_mean_coef=1,
            uniform_flag=1,
            **fixed_uncertainties,
        ),
        Scenario(
            "HighD_LowWh",
            yearly_demand_growth_rate=0.03,
            blue_nile_mean_coef=1,
            white_nile_mean_coef=0.75,
            uniform_flag=1,
            **fixed_uncertainties,
        ),
        Scenario(
            "HighB",
            yearly_demand_growth_rate=0.02,
            blue_nile_mean_coef=1.25,
            white_nile_mean_coef=1,
            uniform_flag=1,
            **fixed_uncertainties,
        ),
    ]
    policy_df = pd.read_csv(f"{output_directory}baseline_results.csv")
    my_policies = [
        Policy(f"Policy{i}", **(policy_df.iloc[i, 1:165].to_dict()))
        for i in policy_df.index
    ]

    before = datetime.now()

    with MultiprocessingEvaluator(em_model) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(my_scenarios, my_policies)

    after = datetime.now()

    with open(f"{output_directory}time_counter_resimulation.txt", "w") as f:
        f.write(
            f"It took {after-before} time to run re-simulation 5 scenarios {len(my_policies)} policies"
        )
    outcomes = pd.DataFrame.from_dict(outcomes)
    experiments.to_csv(f"{output_directory}experiments_resimulation.csv")
    outcomes.to_csv(f"{output_directory}outcomes_resimulation.csv")
