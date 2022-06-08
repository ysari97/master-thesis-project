# Catchment class

import numpy as np


class Catchment:
    def __init__(self, name):
        # Explanation placeholder
        self.name = name

        data_directory = "../NileData/"
        self.inflow = np.loadtxt(f"{data_directory}Inflow{name}.txt")
