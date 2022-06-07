# Model wrapper
import numpy as np


def model_wrapper(model, **kwargs):
    input = [kwargs["v" + str(i)] for i in range(len(kwargs))]
    (
        egypt_irr,
        egypt_90,
        egypt_low_had,
        sudan_irr,
        sudan_90,
        ethiopia_hydro,
    ) = model.evaluate(np.array(input))
    return egypt_irr, egypt_90, egypt_low_had, sudan_irr, sudan_90, ethiopia_hydro
