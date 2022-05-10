# Irrigation district

import numpy as np

class IrrigationDistrict:
    """
    A class used to represent districts that demand irrigation

    Attributes
    ----------
    name : str
        Lowercase non-spaced name of the district
    demand : np.array
        m3
        Vector of water demand from the district throughout the
        simulation horizon
    

    Methods
    -------
    
    """

    def __init__(self, name):
        # Explanation placeholder
        self.name = name

        data_directory = "../Niledata/"
        self.demand = np.loadtxt(f"{data_directory}IrrDemand{name}.txt")
        self.received_flow = np.empty(0)
        self.squared_deficit = np.empty(0)
        self.normalised_deficit = np.empty(0)