import numpy as np
def model_wrapper(**kwargs):
    input = [kwargs['v' + str(i)] for i in range(len(kwargs))]
    Hydropower, Environment, Irrigation = tuple(kwargs["model"].evaluate(np.array(input)))
    return Hydropower, Environment, Irrigation