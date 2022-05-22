# Model wrapper
import numpy as np

def model_wrapper(model, **kwargs):
    input = [kwargs['v' + str(i)] for i in range(len(kwargs))]
    egypt_irr, sudan_irr, ethiopia_hydro = model.evaluate(np.array(input))
    return egypt_irr, sudan_irr, ethiopia_hydro