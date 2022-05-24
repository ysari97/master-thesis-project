# Script for baseline optimization

import pandas as pd
import numpy as np

import pickle

from ema_workbench import (RealParameter, ScalarOutcome, Constant,
                           Model)
from ema_workbench import MultiprocessingEvaluator, SequentialEvaluator, ema_logging

from data_generation import generate_input_data
from wrapper import model_wrapper

from datetime import datetime

import os
import sys

my_path = sys.path[0]
while (my_path[-1] != "/") and (my_path[-1] != "\\"):
    my_path = my_path[:-1]

sys.path.insert(1, my_path + "Model")

from model_nile import ModelNile

if __name__ == '__main__':


    model_object = ModelNile()
    model_object = generate_input_data(model_object, sim_horizon=20)


    parameter_count = model_object.overarching_policy.get_total_parameter_count()

    em_model = Model('NileProblem', function=model_wrapper)

    em_model.levers = [RealParameter('v' + str(i), 0, 1) for i in range(parameter_count)]

    em_model.constants = [Constant("model", model_object)]

    #specify outcomes
    em_model.outcomes = [ScalarOutcome('egypt_irr', ScalarOutcome.MINIMIZE),
                    ScalarOutcome('sudan_irr', ScalarOutcome.MINIMIZE),
                    ScalarOutcome('ethiopia_hydro', ScalarOutcome.MAXIMIZE)]

    ema_logging.log_to_stderr(ema_logging.INFO)

    before = datetime.now()

    with MultiprocessingEvaluator(em_model) as evaluator:
        results = evaluator.optimize(nfe=100, searchover='levers', logging_freq=1,
        epsilons=[0.1,]*len(em_model.outcomes))

    after = datetime.now()

    print(after-before)

    pickle.dump( results, open( "baseline_results.p", "wb" ) )
